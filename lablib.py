import pandas as pd
import numpy as np
import math as m
from scipy import optimize as sp_opt
from scipy import odr as sp_odr
from IPython.display import display as disp
import sympy as smp
import matplotlib.pyplot as plt
import inspect
import os
import collections
import sys

#
# Use `natsort` module, if present, to sort experiment names
# This will yield "4" - "5" - "41" instead of "4" - "41" - "5"
#

try:
	from natsort import natsorted
except ImportError:
	pass


#
# Math function wrappers for sympy
#
# This allows to use the same expression both for symbolic differentiation and for fitting/plotting:
# - smp.* variants accept symbolic variable objects (used inside compute to differentiate the expression),
# - np.* variants accept arrays (e. g. linspace)).
#
# Using a wrapper allows to use the same lambda both to compute a column with errors and to plot a fitted curve.
#

class LablibWrappers:
	def __init__(self, wrapped):
		self.wrapped = wrapped

	def __getattr__(self, name):
		# first try real attributes
		try:
			return getattr(self.wrapped, name)
		except AttributeError as e:
			real_exception = e

		# try to wrap a sympy/numpy method
		try:
			smp_func = getattr(smp, name)
			np_func = getattr(np, name)
			wrapper = lambda x: smp_func(x) \
			                    if type(x).__module__.split(".")[0] == "sympy" \
			                    else np_func(x)

			# a bit of caching
			self.__dict__[name] = wrapper
			return wrapper
		except AttributeError:
			pass

		# finally, re-raise real exception to tell user that lablib has no such attribute
		raise real_exception

# replace our module with a class that allows us to do a __getattr__() on it
sys.modules[__name__] = LablibWrappers(sys.modules[__name__])

#
# Physical constants
#

g = 9.80665 # m/s^2
k = 1.3806485279 * 10**(-23) # J/K
torr = 133.3224 # Pa
R = 8.314472
at = 98066.5 # Pa
mu_0 = 4 * m.pi * 10**-7 # N/A^2


#
# INTERNALS: pandas formatter magic
#
# Used to make disp() automatically show relative errors in percent notation.
# Works by monkey-patching to_html() method of pandas objects because pandas
# lacks a way to set default formatters globally for all objects.
#

# configure pandas to use fixed-point formatting for floats (e. g. with disp())
pd.set_option("float_format", "{:f}".format)

# creates a formatter function wrapper for a given underlying function
# which will pass a given object as "formatters" to the underlying function
# (aka partial bind in C++).
#
# Takes:
# - @fmt_func (callable): method to operate on, pd.DataFrame.to_html() or equivalent
# - @fmt_map: object to pass to @formatters argument of the @fmt_func callable
#
# Returns:
# - new callable
def __make_formatter(fmt_func, fmt_map):
	return lambda *args, **kwargs: fmt_func(formatters = fmt_map, *args, **kwargs)

# wrap formatter function @fmt_func_name in @obj to pass @fmt_map as "formatters".
# repeated applications are supported, new @fmt_map is merged into previous.
#
# Takes:
# - @obj: object to operate on, pd.DataFrame or equivalent
# - @fmt_func_name: name of method in @obj to wrap
# - @fmt_map: object to pass to @formatters argument of the method to wrap
def __set_formatter(obj, fmt_func_name, fmt_map):
	# get un-wrapped formatter function, previously stored by us in "_orig_%s" field,
	# or store it there right now (if this is first invocation on an object)
	orig_fmt_func_name = "_orig_%s" % fmt_func_name
	orig_fmt_func = obj.__dict__.get(orig_fmt_func_name)

	try:
		orig_fmt_func = obj.__getattr__(orig_fmt_func_name)
	except AttributeError:
		orig_fmt_func = obj.__getattr__(fmt_func_name)
		obj.__setattr__(orig_fmt_func_name, orig_fmt_func)

	# get previous formatters map, previously stored by us in "_fmt_map" field,
	# update it with new values and store it back there
	orig_fmt_map_name = "_fmt_map"
	try:
		orig_fmt_map = obj.__getattr__(orig_fmt_map_name)
		orig_fmt_map.update(fmt_map)
	except AttributeError:
		orig_fmt_map = fmt_map
	obj.__setattr__(orig_fmt_map_name, orig_fmt_map)

	# finally, make a formatter wrapper
	fmt_func = __make_formatter(orig_fmt_func, orig_fmt_map)
	obj.__setattr__(fmt_func_name, fmt_func)

# wrap relevant formatter functions in @df to pass @formatters as "formatters".
# repeated applications are supported, new @formatters is merged into previous.
#
# Takes:
# - @df: object to operate on, pd.DataFrame or equivalent
# - @formatters: object to pass to @formatters argument of the formatters
def __set_formatters(df, formatters):
	__set_formatter(df, "to_html", formatters)

# default formatters for __set_formatters in varlist objects
__varlist_formatters = {
	"ErrorRel": "{:.2%}".format
}


#
# DataFrame manipulation
#
# This library uses two kinds of objects to store experimental data.
# Both are internally pandas' DataFrames, differing only in structure.
#
# "Column" objects are DataFrames where each column represents a variable or
# its experimental error, and each row represents an experimental sample
# (e. g. set of instrument readings taken at the same time).
# The error columns are named "Error_%s", where %s is the related variable.
# The index is integral (i. e. rows are simply numbered from 0).
#
# Accessing a column (as a pandas' Series, convertible into a list):
# object["U"], object["Error_U"], ...
#
# Accessing a sub-column (conversion into a list is important):
# list(object["U"][1:4])
#
# "Varlist" objects are dataframes with two columns -- "Value" and "Error",
# and each row represents a variable with its error.
# This object is intended to store experimental constants, which are same
# for all samples in an experiment (e. g. temperature of the working body,
# or atmospheric pressure at the time of an experiment).
# The index consists of variable names (i. e. rows are named after variables).
#
# Accessing a variable's fields:
# object.Value["T"], object.Error["T"]
#
# Accessing a variable as a whole:
# object.loc["T"]
#

# add a column by @name (str), @values (list) and @errors (list) into a "column" object at @where.
#
# Relative errors column is calculated automatically.
#
# Takes:
# - @where ("column" pd.DataFrame): object to add column to
# - @name (str): name of column to add
# - @values (iterable of numbers): data column
# - @errors (iterable of numbers): corresponding error column
def add_column(where, name, values, errors):
	err_name = "Error_%s" % name
	relerr_name = "ErrorRel_%s" % name

	where[name] = values
	where[err_name] = errors
	where[relerr_name] = where[err_name] / where[name]

	__set_formatters(where, { relerr_name: __varlist_formatters["ErrorRel"] })

# add a column by @name (str) from variables with same @name in @varlists
# (list of "varlist" objects) into a "column" object at @where.
#
# As a convenience measure, @varlists can be a dict of "varlist" objects,
# but then @indices must be a list of keys to that dict.
#
# It is unspecified whether relative errors column is recalculated or inherited.
#
# Takes:
# - @where ("column" pd.DataFrame): object to add column to
# - @name (str): name of column to collect/add
# - @varlists (iterable or mapping of "varlist" pd.DataFrame): collection of varlists to make column from
# - @indices (optional, iterable of keys to @varlists): if present, selects varlists to make column from
def make_column(where, name, varlists, indices = None):
	if indices is not None:
		sources = [varlists[i] for i in indices]
	else:
		sources = varlists

	add_column(where = where,
	           name = name,
	           values = [v.Value[name] for v in sources],
	           errors = [v.Error[name] for v in sources])

# make a "varlist" object from raw arguments to pandas' DataFrame ctor;
# use `varlist()` for an empty varlist.
#
# Relative errors column is not calculated.
#
# Takes:
# - arguments to pd.DataFrame ctor
#
# Returns:
# - "varlist" pd.DataFrame
def varlist(*args, **kwargs):
	df = pd.DataFrame(*args, **kwargs, columns = ["Value", "Error", "ErrorRel"])
	__set_formatters(df, __varlist_formatters)
	return df

# creates a "varlist" object by @names (list), @values (list) and @errors (list).
#
# Relative errors column is calculated automatically.
#
# Takes:
# - names (iterable of str): names of variables to create varlist from
# - values (iterable of numbers): values of variables corresponding to @names
# - errors (iterable of numbers): errors of variables corresponding to @names
#
# Returns:
# - "varlist" pd.DataFrame
def var_many(names, values, errors):
	return varlist({ "Value": values,
	                 "Error": errors,
	                 "ErrorRel": [e/v for e, v in zip(errors, values)] },
	               index = names)

# creates a "varlist" object by single variable @name, @value and @error.
#
# Relative errors column is calculated automatically.
#
# Takes:
# - name (str): name of variable to create varlist from
# - value (number): value of variable @name
# - error (number): error of variable @name
#
# Returns:
# - "varlist" pd.DataFrame
def var(name, value, error):
	return var_many([name], [value], [error])

# adds multiple varlists @args to a varlist @where.
#
# Takes:
# - @ehere ("varlist" pd.DataFrame): object to add variables to
# - @args (vararg of "varlist" pd.DataFrame): objects to read variables from
def add(where, *args):
	for arg in args:
		for k, v in arg.iterrows():
			where.loc[k] = v

# adds multiple varlists @args to an iterable (list) of varlists @targets.
#
# Takes:
# - @where (iterable of "varlist" pd.DataFrame): objects to add variables to
# - @args (vararg of "varlist" pd.DataFrame): objects to read variables from
def add_multi(targets, *args):
	for where in targets:
		add(where, *args)


#
# CSV reading
#
# These functions read CSV files into DataFrame objects.
#
# Most probably, you will use a single function `read_standard_layout()`
# to read a typical experiment data file hierarchy.
#

# reads a CSV file @name, returning a "varlist" or "columns" object as appropriate.
#
# For "varlist" objects, the CSV file must have columns "Value" and "Error" and either
# an unnamed first column or "Name" column of variable names.
#
# For "columns" objects, the CSV file must not parse as a "varlist" object
# (i. e. no "Value" and "Error" columns).
#
# In all cases, relative errors columns are calculated automatically.
#
# Takes:
# - @name (str): path to a CSV file
#
# Returns:
# - "varlist" or "columns" pd.DataFrame
def read_csv(name):
	csv = pd.read_csv(name)
	if "Value" in csv.columns and \
	   "Error" in csv.columns:
		if "Name" in csv.columns:
			# use Name column for index if present
			csv.set_index("Name", drop = True, inplace = True)
			# make the index unnamed
			csv.index.set_names(None, inplace = True)

		# compute ErrorRel for a varlist
		try:
			csv["ErrorRel"] = csv["Error"] / csv["Value"]
		except:
			pass
		# set percentage format for the ErrorRel
		__set_formatters(csv, __varlist_formatters)
	else:
		# compute relative error columns for all matching pairs
		relerr_cols = []

		for err_col in csv.columns:
			# check if this is an error column, check if we have pair
			if not err_col.startswith("Error_"):
				continue
			var_col = err_col[6:]
			if not var_col in csv.columns:
				continue

			try:
				relerr_col = "ErrorRel_%s" % var_col
				relerr = csv[err_col] / csv[var_col]
				# insert the new column after the error column
				csv.insert(list(csv.columns).index(err_col) + 1,
				           relerr_col,
				           relerr)
				# save its name
				relerr_cols.append(relerr_col)
			except:
				pass


		# set percentage format for all these columns
		__set_formatters(csv, { k: __varlist_formatters["ErrorRel"]
		                        for k in relerr_cols })

	return csv

# reads a CSV file @name, returning an empty "varlist" object in failure case.
#
# NOTE: a "varlist" object is not a "columns" object.
#
# Takes:
# - @name (str): path to a CSV file
#
# Returns:
# - "varlist" or "columns" pd.DataFrame
def maybe_read_csv(name):
	try:
		return read_csv(name)
	except OSError:
		return varlist()

# reads a directory of CSV files @name, returning a dictionary of appropriate objects
# or an empty dictionary in failure case.
#
# Takes:
# - @name (str): path to a directory containing CSV files
#
# Returns:
# - dictionary of "varlist" or "columns" pd.DataFrame
def maybe_read_csv_dir(name):
	ret = {}

	try:
		filelist = os.listdir(name)
	except OSError:
		return ret

	for f in filelist:
		if f.endswith(".csv"):
			e = f[:-4]
			ret[e] = read_csv(os.path.join(name, f))

	return ret

# reads a typical experiment file hierarchy at current directory.
#
# Hierarchy:
# ./constants.csv              global constants for all experiments
# ./constants/                 a directory
# ./constants/$name.csv        per-experiment constants for experiment $name
# ./measurements/              a directory
# ./measurements/$name.csv     per-experiment samples for experiment $name
#
# Return: data, columns, experiments
# - experiments (list of str):          list of experiment names, used as keys in the below dicts
# - columns (dict of "column" objects): [$name]: per-experiment samples of $name
# - data (dict of "varlist" objects):   [$name]: global + per-experiment constants of $name
#                                       ["global"]: just global constants
#
def read_standard_layout():
	constants = maybe_read_csv("constants.csv")
	data = maybe_read_csv_dir("constants")
	columns = maybe_read_csv_dir("measurements")

	experiments = set(list(columns.keys()) + list(data.keys()))

	if 'natsorted' in globals():
		experiments = natsorted(experiments)
	else:
		experiments = sorted(experiments)

	for e in experiments:
		d = varlist()
		add(d, constants)
		if e in data:
			add(d, data[e])
		data[e] = d

		if not e in columns:
			columns[e] = pd.DataFrame()

	# also add "global" varlist with only the global constants
	d = varlist()
	add(d, constants)
	data["global"] = d
	columns["global"] = pd.DataFrame()

	return data, columns, experiments

#
# Curve fitting
#
# These functions fit curves to experimental datasets using a variant
# of the least squares method.
#
# The functions provide a varying level of abstraction.
#

# fit(): most basic fitting function.
#
# Takes:
# - @name (str): name of the fitting session (used purely for logging)
# - @model (callable): a callable object representing the curve: `model(x, parameters...) -> y`
# - @model_args (list of str): names of parameters for the model
# - @x, @y, @xerr, @yerr (iterable of float): the data and error columns
# - @initial (list of float): initial guesses for the ODR solver
# - @prefit (bool): whether to perform OLS pass to get initial guesses for the ODR solver
# - @noop (bool): whether to skip fitting and return initial guess or all-zeros
#
# Returns:
# - "varlist" pd.DataFrame of found parameters
def fit(name, model, model_args, x, y, xerr, yerr, initial = None, prefit = False, noop = False):
	if noop:
		return var_many(names = model_args,
		                values = initial if initial is not None else [0 for x in model_args],
		                errors = [0 for x in model_args])

	# use OLS (ordinary least squares) to find initial guesses
	if prefit or initial is None:
		beta, cov = sp_opt.curve_fit(model,
		                             xdata = x,
		                             ydata = y,
		                             sigma = yerr,
		                             absolute_sigma = True,
		                             maxfev = int(1e6),
		                             p0 = initial)

		fit_result = var_many(names = model_args,
		                      values = beta,
		                      errors = [ cov[i, i]**0.5
		                                 for i, v
		                                 in enumerate(cov) ])

		print("Initial guesses for %s:\n%s\n" % (name, fit_result))
		initial = beta

	# use ODR (Deming regression) which is a special case of TLS (total least squares)
	# to find results accounting for both X and Y uncertainties
	odr_model = sp_odr.Model(lambda B, x: model(x, *B))
	odr_data = sp_odr.RealData(x = x, y = y, sx = xerr, sy = yerr)
	odr = sp_odr.ODR(odr_data, odr_model, beta0 = initial, maxit = int(1e6))
	odr_output = odr.run()

	fit_result = var_many(names = model_args, values = odr_output.beta, errors = odr_output.sd_beta)
	print("Final guesses for %s:" % name)
	disp(fit_result)

	return fit_result

# fit2(): fitting function with introspection of model arguments.
# The parameter names are inferred from the @model argument names (by Python runtime introspection).
# The results (found parameters) are added to @data under corresponding names.
#
# Takes:
# - @name (str): name of the fitting session (used purely for logging)
# - @model (callable): a callable object representing the curve: `model(x, parameters...) -> y`
# - @x, @y, @xerr, @yerr (iterable of float): the data and error columns
# - @data ("varlist" pd.DataFrame): an object to store results to
# - @initial (list of float): initial guesses for the ODR solver
# - @prefit (bool): whether to perform OLS pass to get initial guesses for the ODR solver
# - @noop (bool): whether to skip fitting and return initial guess or all-zeros
#
# Returns:
# - input model callable with substituted parameters
def fit2(name, model, x, y, xerr, yerr, data, initial = None, **kwargs):
	model_args = list(inspect.signature(model).parameters.keys())[1:]
	result = fit(name, model, model_args, x, y, xerr, yerr, initial, **kwargs)
	add(data, result)

	return lambda x: model(x, *[data.Value[a] for a in model_args])


# plotfit() implementation detail: converts input arguments to batch form, expanding singles.
#
# NOTE: internal function.
#
# Takes:
# - multiple (argument name, argument, checker function) tuples
#
# Returns:
# - a list of normalized arguments in the input order
def plotfit_normalize_batch(*args):
	# detect single and batched arguments as well as the batch length
	batch_length = 1
	batch_length_argname = None
	batch_is_single = []
	for name, arg, checker in args:
		# explicitly check for valid single argument because an str is an iterable of str (incredibly)
		if checker(arg):
			batch_is_single += [True]
		# only then check for valid batched argument
		elif is_iter(arg, check0 = checker):
			if batch_length == 1:
				batch_length = len(arg)
				batch_length_argname = name
			elif batch_length != len(arg):
				raise ValueError("plotfit: batch arguments of different length: `%s` (%d) != `%s` (%d)"
				                 % (batch_length_argname, batch_length, name, len(arg)))
			batch_is_single += [False]
		else:
			raise ValueError("Not a valid argument nor a batch thereof: `%s` == %s" % (name, repr(arg)))

	# actually normalize arguments
	batch_args = []
	for (name, arg, checker), is_single in zip(args, batch_is_single):
		if is_single:
			if batch_length > 1 and arg is not None and (name == "data" or name == "label"):
				print("plotfit: Warning: `%s` is not a batch -- this is probably not what you want" % name)
			batch_args += [ [arg] * batch_length ]
		else:
			batch_args += [ arg ]

	return batch_args

# plotfit() implementation detail: converts column input arguments to the simplest "direct" form.
#
# NOTE: internal function.
#
# Takes:
# - columns: the `columns` input argument (batch of pd.DataFrame)
# - args: multiple (data column name, data column arg, error column name, error column arg) tuples
#
# Returns:
# - list of data columns in direct form (in input order)
# - list of corresponding error columns in direct form (in input order)
# - list of deduced data column names (in input order)
def plotfit_normalize_form(columns, *args):
	direct_args = []
	direct_errs = []
	direct_names = []

	# we'd rather use `columns` as iteration variable
	b_columns = columns

	for n_arg, b_arg, n_err, b_err in args:
		direct_b_arg = []
		direct_b_err = []
		direct_b_name = []

		for arg, err, columns in zip(b_arg, b_err, b_columns):
			direct_arg = None
			direct_err = None
			direct_name = None

			# process the data column first...
			if is_synthesized_column(arg):
				direct_name = arg[0].name
				direct_arg = [obj.Value for obj in arg]
				if err is True:
					direct_err = [obj.Error for obj in arg]
			elif is_symbolic_column(arg):
				if columns is None:
					raise ValueError("plotfit: symbolic columns (in `%s`) only allowed if 'columns' are provided"
					                 % n_arg)
				direct_name = arg
				direct_arg = columns[arg]
				if err is True:
					err_arg = "Error_%s" % arg
					if err_arg in columns:
						direct_err = columns[err_arg]
			else:
				assert(is_direct_column(arg))
				direct_name = arg.name if is_series(arg) else "<%s>" % n_arg
				direct_arg = arg

			# then process the explicit error column
			if is_symbolic_column(err):
				if columns is None:
					raise ValueError("plotfit: symbolic columns (in `%s`) only allowed if 'columns' are provided"
					                 % n_err)
				direct_err = columns[err]
			elif is_direct_column(err):
				direct_err = err
			else:
				assert(err is None or err is True)
				pass

			direct_b_arg += [ direct_arg ]
			direct_b_err += [ direct_err ]
			direct_b_name += [ direct_name ]

		direct_args += [ direct_b_arg ]
		direct_errs += [ direct_b_err ]
		direct_names += [ direct_b_name ]

	return direct_args, direct_errs, direct_names

# fits `y` against `x` using `model`, storing results to `data`, and plots the dataset
# together with the fitted curve.
#
# This function supports a multitude of input formats; for details see in-line
# comments below.
#
# Arguments @title, @xlabel, @ylabel and @axis translate into corresponding
# plt.xxx() calls and exist for convenience. If not specified or None, the
# corresponding calls are skipped.
#
# The @x, @y, @xerr and @yerr parameters specify data and error columns.
# They can be expressed in multiple forms:
# - "synthesized columns": a column is formed using sequences of individual
#   variables (varlist rows)
#   - each of @x, @y can be
#     - a sequence of pd.Series containing Value and Error in index
#   - Example:
#
#     plotfit(x = [data[e].loc["T"] for e in experiments],
#             y = [data[e].loc["phi"] for e in experiments])
#
# - "symbolic columns": a column is selected by name from the @columns pd.DataFrame
#   - each of @x, @y, @xerr, @yerr can be
#     - a string (str) that is interpreted as a column name
#   - @columns is
#     - a "columns" pd.DataFrame
#   - Example:
#
#     plotfit(x = "T",
#             y = "phi",
#             columns = columns[e][5:10])
#
# - "legacy": a column is passed directly as a sequence of values
#   - each of @x, @y, @xerr, @yerr can be
#     - a sequence of values of any form
#   - Example:
#
#     plotfit(x = columns[e]["T"][5:10],
#             xerr = columns[e]["Error_T"][5:10],
#             y = columns[e]["phi_1"][5:10],
#             yerr = [phi_1 * 0.001 for phi_1 in columns[e]["phi_1"][5:10]])
#
# The error columns (@xerr and @yerr) can also be specified as:
# - None, in which case zero errors are assumed;
# - True, in which case the errors are inferred from the data column specification:
#   - if the data column is specified symbolically as "foo", the error column
#     is taken to be "Error_foo";
#   - if the data column is specified as a sequence of varlist rows, the errors
#     are extracted from these rows (which is the suggested mode of operation).
#
# If the symbolic form is used, the @columns argument must be a "columns" pd.DataFrame
# holding the actual dataset. Otherwise it can be omitted or None.
#
# Furthermore, the @fit_columns argument overrides @columns for purposes of fitting.
# This feature can be used to fit against a "good" subset of experimental data
# while plotting both bad and good data. If not specified or None, this feature
# is disabled.
#
# Finally, the @label argument sets the label of this dataset and the corresponding
# curve on the plot legend. If not specified or None, the dataset is not added
# to the legend.
#
# Each data argument above (@x, @y, @xerr, @yerr, @columns, @fit_columns, @label)
# additionally can accept a sequence of above-described values, called a "batch".
# All batches must have the same length (that is, it is incorrect to pass two
# @x columns and three @y columns).
# Example:
#
#     plotfit(x = [columns[e]["T"] for e in experiments],
#             y = [columns[e]["phi_1"] for e in experiments])
#
#     plotfit(x = "T",
#             y = "phi_1",
#             columns = [columns[e] for e in experiments])
#
# Mixing batches and non-batches is allowed. In this case each non-batched argument
# will be replicated. For example, you can use a single error column for a batch
# of curves, or plot multiple @y columns against the same @x column:
#
#     plotfit(x = "T",
#             y = ["phi_1", "phi_2"],
#             columns = columns[e])
#
# The @model argument must be a callable (function or lambda) where the first
# argument is the model argument ("x") and the remaining arguments are model parameters.
# The model parameter names are significant. If not specified or None, fitting and
# curve plotting steps are skipped.
#
# Correspondingly, the @data argument must be a "varlist" pd.DataFrame. It is used
# to store the results of the fitting step. Found parameter values are stored with
# their names as in Python source! If not specified or None, fitting and curve
# plotting steps are skipped.
#
# The arguments @fit_args, @plot1_args, @plot2_args and @linspace_args can be used
# to provide additional arguments to back-end fitting and plotting functions.
# If not specified, they default to an empty dict. NB: if explicitly set to None,
# the corresponding steps are skipped!
#
# Takes:
# - @title (str or None),
#   @xlabel (str or None),
#   @ylabel (str or None),
#   @axis (list of 4 numbers or None): convenience arguments for setting up a plot
# - @model (callable or None): a callable object representing the curve: `model(x, parameters...) -> y`
# - @x, @y (pd.Series or str or iterable of varlist rows or sequence thereof): the data columns
# - @xerr, @yerr (as in @x/@y or True or None or sequence thereof): the corresponding error columns
# - @data ("varlist" pd.DataFrame or None): an object to store results to
# - @columns ("columns" pd.DataFrame or sequence thereof or None): the data source for input columns
# - @fit_columns ("columns" pd.DataFrame or sequence thereof or None): alternative @columns (probably subset) to use for fitting
# - @label (str or sequence thereof or None): a label for the plot legend
# - @fit_args (dict or None): additional keyword arguments to ll.fit2()
# - @plot1_args (dict or None): additional keyword arguments to plt.errorbar() when plotting experimental data
# - @plot2_args (dict or None): additional keyword arguments to plt.plot() when plotting fitted curves
# - @linspace_args (dict): additional keyword arguments to ll.linspace()
#
# Returns:
# - list of input model callable with substituted arguments
def plotfit(*,
            title = None, xlabel = None, ylabel = None, model = None, axis = [None, None, None, None],
            x, xerr = True, y, yerr = True, data = None, columns = None, fit_columns = None, label = None,
            fit_args = {}, plot1_args = {}, plot2_args = {}, linspace_args = {}):

	have_fit_columns = (fit_columns is not None and fit_columns is not columns)

	#
	# First, normalize input data. Begin with detecting batches and replicating non-batched input.
	#

	# Note the renaming: we'd rather use bare words as iteration variables
	(o_x,
	 o_y,
	 o_xerr,
	 o_yerr,
	 b_columns,
	 b_fit_columns,
	 b_data,
	 b_label) = plotfit_normalize_batch(("x", x, is_valid_input),
	                                    ("y", y, is_valid_input),
	                                    ("xerr", xerr, is_valid_input_err),
	                                    ("yerr", yerr, is_valid_input_err),
	                                    ("columns", columns, is_valid_columns),
	                                    ("fit_columns", fit_columns, is_valid_columns),
	                                    ("data", data, is_valid_data),
	                                    ("label", label, is_valid_label))
	batch_size = len(o_x)

	#
	# Now convert arguments to direct form.
	#

	# first for normal b_columns...
	((b_x, b_y),
	 (b_xerr, b_yerr),
	 (b_x_name, b_y_name)) = plotfit_normalize_form(b_columns,
	                                                ("x", o_x, "xerr", o_xerr),
	                                                ("y", o_y, "yerr", o_yerr))

	# ...then for b_fit_columns (defaulting to b_columns)
	if have_fit_columns:
		b_fit_columns = [ f_c if f_c is not None else c
		                  for c, f_c in zip(b_columns, b_fit_columns) ]

		((b_fit_x, b_fit_y),
		 (b_fit_xerr, b_fit_yerr),
		 _                   ) = plotfit_normalize_form(b_fit_columns,
		                                                ("x", o_x, "xerr", o_xerr),
		                                                ("y", o_y, "yerr", o_yerr))
	else:
		(b_fit_x,
		 b_fit_y,
		 b_fit_xerr,
		 b_fit_yerr) = (b_x,
		                 b_y,
		                 b_xerr,
		                 b_yerr)

	#
	# Now do the fitting.
	#

	if model is not None and data is not None and fit_args is not None:
		b_model = []
		for (x, y,
		     xerr, yerr,
		     x_name, y_name,
		     data, label) in zip(b_fit_x, b_fit_y,
			                 b_fit_xerr, b_fit_yerr,
			                 b_x_name, b_y_name,
			                 b_data, b_label):
			fit_name = "%s = f(%s)" % (y_name, x_name)
			if label is not None:
				fit_name += " [label = %s]" % label
			fit_model = fit2(name = fit_name, model = model,
			                 x = x, y = y, xerr = xerr, yerr = yerr,
			                 data = data, **fit_args)
			b_model += [ fit_model ]
	else:
		b_model = None

	#
	# Now do the drawing.
	# First draw the experimental dataset, then (if not disabled) perform
	# fitting and draw the curves.
	#

	if title is not None:
		plt.title(title)
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)

	# if full dataset != fit dataset, then plot the full dataset in reduced intensity
	if plot1_args is not None and have_fit_columns:
		plot1_dfl_args = {
		    "linestyle": "None",
		    "marker": ".",
		    "alpha": 0.5,
		    "markersize": 0.5,
		    "linewidth": 0.5,
		    "markeredgewidth": 0.5,
		}
		plot1_dfl_args.update(plot1_args)

		for (x, y,
		     xerr, yerr) in zip(b_x, b_y,
		                        b_xerr, b_yerr):
			# no need to have another legend entry for the dim-colored
			# points, they are in the same color as the fit dataset
			plt.errorbar(x = x, y = y, xerr = xerr, yerr = yerr, label = None,
			             **plot1_dfl_args)

		# mandatorily reset colors so that parts of the same dataset
		# are plotted in the same color
		plt.gca().set_prop_cycle(None)

	# then, plot the fit dataset in full intensity
	if plot1_args is not None:
		plot1_dfl_args = {
		    "linestyle": "None",
		    "marker": ".",
		}
		plot1_dfl_args.update(plot1_args)

		has_labels = False
		for (x, y,
		     xerr, yerr,
		     label) in zip(b_fit_x, b_fit_y,
		                   b_fit_xerr, b_fit_yerr,
		                   b_label):
			if label is not None:
				has_labels = True
			plt.errorbar(x = x, y = y, xerr = xerr, yerr = yerr, label = label,
			             **plot1_dfl_args)

		# if we are plotting a batch, reset colors again so that fitted curves
		# will match their data points.  otherwise, it is better to have them
		# in different colors.
		if batch_size > 1:
			plt.gca().set_prop_cycle(None)

	if b_model is not None and plot2_args is not None:
		for (x, fit_model, label) in zip(b_x, b_model, b_label):
			if batch_size > 1:
				# no need to have another legend entry for the fitted curve,
				# it's in the same color as the dataset
				label = None
			else:
				if label is not None:
					label = "Fit for %s" % label
			fit_linspace = linspace(x, **linspace_args)
			plt.plot(fit_linspace, fit_model(fit_linspace), label = label,
			         **plot2_args)

	if has_labels:
		plt.legend(loc = "best")
	plt.axis(axis)
	return b_model

#
# Utility functions
#

# minmax(): returns minimum and maximum of the input sequence.
def minmax(arg):
	return min(arg), max(arg)

# linspace(): returns a np.linspace with spanning range calculated from an input sequence.
#
# Takes:
# - @arg (sequence of numbers): the set to span
# - @ticks (number): amount of evenly spaced numbers to return
# - @pre (number): how much to span before the range
# - @post (number): how much to span before the range
#
# Returns:
# - np.linspace
def linspace(arg, ticks = 100, pre = 0, post = 0.1):
	min_arg, max_arg = minmax(arg)
	return np.linspace(min_arg - pre * (max_arg - min_arg),
	                   max_arg + post * (max_arg - min_arg),
	                   ticks)


# is_number(): checks if @arg is castable to float.
def is_number(arg):
	try:
		arg = float(arg)
		return True
	except:
		return False

# is_iter(): checks if @arg is an iterable.
def is_iter(arg, check0 = None):
	try:
		if not isinstance(arg, collections.abc.Iterable):
			return False
		if check0 is not None:
			for obj in arg:
				return check0(obj)
		return True
	except:
		return False

# is_str(): checks if @arg is a string.
def is_str(arg):
	return isinstance(arg, str)

# is_series(): checks if @arg is a pd.Series.
def is_series(arg):
	return isinstance(arg, pd.Series)

# is_dataframe(): checks if @arg is a pd.DataFrame.
def is_dataframe(arg):
	return isinstance(arg, pd.DataFrame)

# is_var(): checks if @arg is a row of "varlist" pd.DataFrame.
def is_var(arg):
	return (is_series(arg) and
	        "Value" in arg and
	        "Error" in arg)

# plotfit() implementation detail: checks if @arg is a valid synthesized column specification.
def is_synthesized_column(arg):
	return is_iter(arg, check0 = is_var)

# plotfit() implementation detail: checks if @arg is a valid symbolic column specification.
def is_symbolic_column(arg):
	return isinstance(arg, str)

# plotfit() implementation detail: checks if @arg is a valid direct column specification.
def is_direct_column(arg):
	return is_iter(arg, check0 = is_number)

# plotfit() implementation detail: checks if @arg is a valid data column input.
def is_valid_input(arg):
	return (is_synthesized_column(arg) or
	        is_symbolic_column(arg) or
	        is_direct_column(arg))

# plotfit() implementation detail: checks if @arg is a valid error column input.
def is_valid_input_err(arg):
	return (arg is None or
	        arg is True or
	        # no synthesized columns
	        is_symbolic_column(arg) or
	        is_direct_column(arg))

# plotfit() implementation detail: checks if @arg is a valid "columns" pd.DataFrame object.
def is_valid_columns(arg):
	return (arg is None or
	        is_dataframe(arg))

# plotfit() implementation detail: checks if @arg is a valid "varlist" pd.DataFrame object.
def is_valid_data(arg):
	return (arg is None or
	        (is_dataframe(arg) and
	         "Value" in arg and
	         "Error" in arg))

# plotfit() implementation detail: checks if @arg is a valid dataset label.
def is_valid_label(arg):
	return (arg is None or
	        is_str(arg))

#
# General form expression computation with errors (uncertainties).
#

# sym_error(): computes error of given expression (symbolically) given list of
# its variables (to consider in the calculation).
#
# NOTE: internal function.
#
# Returns:
# - sympy expression representing error of the expression
# - list of sympy variables representing errors of given variables
# - list of sympy expressions representing derivatives of given variables
# - list of sympy expressions representing (error*derivative)^2 of given variables
def sym_error(expr, expr_vars):
	expr_err_vars = []
	expr_err_e_d_sq = []
	expr_err_derivs = []
	for var in expr_vars:
		err_var = smp.symbols("Error_%s" % var.name)
		err_deriv = smp.diff(expr, var)
		expr_err_vars += [err_var]
		expr_err_derivs += [err_deriv]
		expr_err_e_d_sq += [(err_deriv*err_var)**2]
	return smp.sqrt(sum(expr_err_e_d_sq)), expr_err_vars, expr_err_derivs, expr_err_e_d_sq

# computes a substitution dictionary for the .subs() method of the symbolic expression.
#
# NOTE: internal function
#
# Takes:
# - @expr_vars (sequence of sympy variables): list of variables to substitute
# - @expr_err_vars (sequence of sympy variables): list of variables representing errors of @expr_vars
# - @data ("varlist" pd.DataFrame): values of the variables to substitute
#
# Returns:
# - dictionary of "sympy variable -> substitution value" for all input sympy variables
def sym_make_subs(expr_vars, expr_err_vars, data):
	var_pairs = { var: data.Value[var.name]
	              for var
	              in expr_vars
	              if var.name in data.Value }
	err_pairs = { err_var: data.Error[var.name]
	              for var, err_var
	              in zip(expr_vars, expr_err_vars)
	              if var.name in data.Error }

	var_pairs.update(err_pairs)
	return var_pairs

# shows (using IPython.display.display()) verbose information about errors of variables in an expression
# and their influence.
#
# NOTE: internal function.
def sym_compute_show_error_influences(name, data, expr_subs, expr_vars, expr_err_derivs, expr_err_e_d_sq):
	bits = pd.DataFrame({ var.name: { "Error": data.Error[var.name] if var.name in data.Error else None,
	                                  "Derivative": deriv.subs(expr_subs),
	                                  "(E*D)^2": e_d_sq.subs(expr_subs) }
	                      for var, deriv, e_d_sq in zip(expr_vars, expr_err_derivs, expr_err_e_d_sq) },
	                     index = ["Error", "Derivative", "(E*D)^2"]).T

	# try to sort error influences (if any variables are unresolved, this will fail)
	try:
		bits = bits.sort_values("(E*D)^2", ascending=False)
	except:
		pass

	if name:
		print("Error influence estimations for %s:" % name)
	else:
		print("Error influence estimations:")
	disp(bits)

# computes a symbolic expression along with its error from given data.
#
# NOTE: obsolete function.
def sym_compute(name, expr, data):
	expr_vars = expr.atoms(smp.Symbol)
	expr_err, expr_err_vars, expr_err_derivs, expr_err_e_d_sq = sym_error(expr, expr_vars)
	expr_subs = sym_make_subs(expr_vars, expr_err_vars, data)

	sym_compute_show_error_influences(name,
	                                  data,
	                                  expr_subs,
	                                  expr_vars,
	                                  expr_err_derivs,
	                                  expr_err_e_d_sq)

	return var(name, float(expr.subs(expr_subs)), float(expr_err.subs(expr_subs)))

# computes a substitution dictionary (template) for the .subs() method of the symbolic expression
# with column names instead of values.
#
# NOTE: internal, obsolete function.
def sym_make_subs_cols_mapping(expr_vars, expr_err_vars, cols_mapping):
	var_pairs = { var: cols_mapping[var.name]["Value"]
	              for var
	              in expr_vars
	              if var.name in cols_mapping }
	err_pairs = { err_var: cols_mapping[var.name]["Error"]
	              for var, err_var
	              in zip(expr_vars, expr_err_vars)
	              if var.name in cols_mapping }

	var_pairs.update(err_pairs)
	return var_pairs

# computes a substitution dictionary (template) for the .subs() method of the symbolic expression
# with column names instead of values.
# Generates default column names by convention rather than by mapping.
#
# NOTE: internal function.
def sym_make_subs_cols_mapping_infer(expr_vars, expr_err_vars, cols):
	var_pairs = { var: var.name
	              for var
	              in expr_vars
	              if var.name in cols }
	err_pairs = { err_var: "Error_%s" % var.name
	              for var, err_var
	              in zip(expr_vars, expr_err_vars)
	              if "Error_%s" % var.name in cols }

	var_pairs.update(err_pairs)
	return var_pairs

# computes a symbolic expression along with its error from given column of data.
#
# NOTE: obsolete function.
#
# data is a DataFrame with common constants for all instances of the computation
# cols is a DataFrame with columns
# cols_mapping is formatted as a dictionary of dict[<var>]["Error", "Value"] = <column name>
#              if None, columns are matched by names (errors as Error_<var>)
def sym_compute_column(name, expr, data, cols_mapping, cols):
	expr_vars = expr.atoms(smp.Symbol)
	expr_err, expr_err_vars, expr_err_derivs, expr_err_e_d_sq = sym_error(expr, expr_vars)
	expr_subs = sym_make_subs(expr_vars, expr_err_vars, data)

	sym_compute_show_error_influences(name,
	                                  data,
	                                  expr_subs,
	                                  expr_vars,
	                                  expr_err_derivs,
	                                  expr_err_e_d_sq)

	# pre-substitute constants
	expr = expr.subs(expr_subs)
	expr_err = expr_err.subs(expr_subs)

	if cols_mapping is not None:
		expr_subs_cols_mapping = sym_make_subs_cols_mapping(expr_vars,
		                                                    expr_err_vars,
		                                                    cols_mapping)
	else:
		expr_subs_cols_mapping = sym_make_subs_cols_mapping_infer(expr_vars,
		                                                          expr_err_vars,
		                                                          cols)

	expr_subs_column = [ { var: s[col_name]
	                       for var, col_name
	                       in expr_subs_cols_mapping.items() }
	                     for i, s
	                     in cols.iterrows() ]

	expr_column = [ float(expr.subs(data))
	                for data
	                in expr_subs_column ]
	expr_err_column = [ float(expr_err.subs(data))
	                    for data
	                    in expr_subs_column ]

	return expr_column, expr_err_column

# computes a substitution dictionary for the .subs() method of the symbolic expression.
#
# NOTE: internal function.
#
# Takes:
# - @expr_vars (iterable of sympy variables): list of variables to substitute
# - @expr_err_vars (iterable of sympy variables): list of variables representing errors of @expr_vars
# - @aux (mapping of str to mapping of "Value", "Error" to number or str): values of the variables to substitute
#                                                                          (non-numbers are ignored)
def sym_make_subs_aux(expr_vars, expr_err_vars, aux):
	var_pairs = { var: aux[var.name]["Value"]
	              for var
	              in expr_vars
	              if var.name in aux
	              and "Value" in aux[var.name]
	              and is_number(aux[var.name]["Value"]) }

	err_pairs = { err_var: aux[var.name]["Error"]
	              for var, err_var
	              in zip(expr_vars, expr_err_vars)
	              if var.name in aux
	              and "Error" in aux[var.name]
	              and is_number(aux[var.name]["Error"]) }

	var_pairs.update(err_pairs)
	return var_pairs

# computes a substitution dictionary template for the .subs() method of the symbolic expression
# with column names instead of values.
# For error columns, both default names and the explicit mapping are considered.
#
# NOTE: internal function.
#
# Takes:
# - @expr_vars (iterable of sympy variables): list of variables to substitute
# - @expr_err_vars (iterable of sympy variables): list of variables representing errors of @expr_vars
# - @cols ("columns" pd.DataFrame): available columns for substitution
# - @aux (mapping of str to mapping of "Value", "Error" to number or str): column names to substitute
#                                                                          (non-str are ignored)
def sym_make_subs_cols_meta(expr_vars, expr_err_vars, cols, aux):
	if cols is None:
		return {}

	# first, find columns with matching names
	var_pairs_inferred = sym_make_subs_cols_mapping_infer(expr_vars, expr_err_vars, cols)
	if aux is None:
		return var_pairs_inferred

	# then build substitutions for column references in aux
	var_pairs = { var: aux[var.name]["Value"]
	              for var
	              in expr_vars
	              if var.name in aux
	              and "Value" in aux[var.name]
	              and isinstance(aux[var.name]["Value"], str)
		      and aux[var.name]["Value"] in cols }

	err_pairs = { err_var: aux[var.name]["Error"]
	              for var, err_var
	              in zip(expr_vars, expr_err_vars)
	              if var.name in aux
	              and "Error" in aux[var.name]
	              and isinstance(aux[var.name]["Error"], str)
		      and aux[var.name]["Error"] in cols }

	var_pairs.update(err_pairs)

	# then merge everything, prioritising explicit statements
	var_pairs_inferred.update(var_pairs)
	return var_pairs_inferred

# This is the new general interface for computing data along with its error from given dataset.
#
# Takes:
# - @name (str): the result variable name
# - @expr (sympy expression or callable): expression to compute
#   (if @expr is a function, then its argument names are significant)
# - @data ("varlist" pd.DataFrame): variables to be used for all instances of computation
# - @cols ("columns" pd.DataFrame): variables to be substituted row-by-row
# - @aux (mapping of str to mapping of "Value", "Error" to number or str): explicit mappings
#   (strings mean column names)
#
# Returns:
# - the sympified expression (valuable if a function is passed, but the expression is needed afterwards)
# - list of its sympy variables
# - list of its errors as sympy variables (in the same order)
#
# The default name for error columns (in absence of mapping) is "Error_<var>".
def compute(name, expr, data, columns = None, aux = None, debug = False, expr_args = None):
	if type(expr).__name__ == "function":
		expr_fn = expr
		if expr_args is None:
			expr_args = list(inspect.signature(expr_fn).parameters.keys())
		expr_vars = smp.symbols(expr_args)
		expr = expr_fn(*expr_vars)
	else:
		expr_vars = expr.atoms(smp.Symbol)

	expr_err, expr_err_vars, expr_err_derivs, expr_err_e_d_sq = sym_error(expr, expr_vars)

	# front check if we have enough data
	for v in expr_vars:
		if not (v.name in data.Value or
		        v.name in columns or
			(aux is not None and v.name in aux and "Value" in aux[v.name])):
			raise IndexError("Variable %s does not exist in dataset" % v.name)

		if not (v.name in data.Error or
		        "Error_%s" % v.name in columns or
			(aux is not None and v.name in aux and "Error" in aux[v.name])):
			raise IndexError("Error for variable %s does not exist in dataset" % v.name)

	# build substitutions from data
	expr_subs = sym_make_subs(expr_vars, expr_err_vars, data)

	# build substitutions from immediate values in cols_mapping
	if aux is not None:
		expr_subs_aux = sym_make_subs_aux(expr_vars, expr_err_vars, aux)
		expr_subs.update(expr_subs_aux)

	# pre-substitute constants
	expr = expr.subs(expr_subs)
	expr_err = expr_err.subs(expr_subs)

	# build substitution template from columns and column references in cols_mapping
	expr_subs_cols = sym_make_subs_cols_meta(expr_vars, expr_err_vars, columns, aux)

	# show some verbose information
	if debug:
		if expr_subs_cols:
			print("Computing row of %s" % name)
		else:
			print("Computing variable %s" % name)

		sym_compute_show_error_influences(None,
		                                  data,
		                                  expr_subs,
		                                  expr_vars,
		                                  expr_err_derivs,
		                                  expr_err_e_d_sq)

	# if any variables are given in the form of columns, then we are computing
	# a column, otherwise -- a single variable
	if expr_subs_cols:
		expr_subs_rows = [ { var: row[col_name]
		                     for var, col_name
		                     in expr_subs_cols.items() }
		                   for i, row
		                   in columns.iterrows() ]

		expr_rows = [ float(expr.subs(data))
		              for data
		              in expr_subs_rows ]
		expr_err_rows = [ float(expr_err.subs(data))
		                  for data
		                  in expr_subs_rows ]

		if debug:
			print("(Not displaying column '%s' of N=%d rows)" % (name, len(expr_rows)))

		add_column(columns,
		           name = name,
		           values = expr_rows,
		           errors = expr_err_rows)

	else:
		expr_out = var(name, float(expr), float(expr_err))

		if debug:
			print("Result:")
			disp(expr_out)

		add(data, expr_out)

	return expr, expr_vars, expr_err_vars

# Apply a function to multiple columns or single variables in-place.
def convert(names, expr, data = None, columns = None):
	for n in names:
		err_n = "Error_%s" % n

		if n in data.Value:
			add(data,
			    var(n,
			        expr(data.Value[n]),
			        expr(data.Error[n])))

		elif n in columns and err_n in columns:
			add_column(columns,
			           name = n,
			           values = [ expr(v) for v in columns[n] ],
			           errors = [ expr(v) for v in columns[err_n] ])

		elif n in columns:
			columns[n] = [ expr(v) for v in columns[n] ]

		else:
			raise IndexError("Variable %s does not exist in dataset" % n)

#  vim: set ts=8 sw=8 tw=0 noet ft=python :
