import pandas as pd
import numpy as np
from scipy import optimize as sp_opt
from scipy import odr as sp_odr
from IPython.display import display as disp
import sympy as smp
import inspect
import os

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

# TODO: generalize math wrappers
def exp(x):
	if type(x).__module__.split(".")[0] == "sympy":
		return smp.exp(x)
	else:
		return np.exp(x)

def log(x):
	if type(x).__module__.split(".")[0] == "sympy":
		return smp.log(x)
	else:
		return np.log(x)


#
# Physical constants
#

g = 9.80665 # m/s^2
k = 1.3806485279 * 10**(-23) # J/K
torr = 133.3224 # Pa
R = 8.314472
at = 98066.5 # Pa


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
def __make_formatter(fmt_func, fmt_map):
	return lambda *args, **kwargs: fmt_func(formatters = fmt_map, *args, **kwargs)

# wrap formatter function @fmt_func_name in @obj to pass @fmt_map as "formatters".
# repeated applications are supported, new @fmt_map is merged into previous.
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

# add a column by @name (str), @values (list) and @errors (list) into a "column" object at @where
#
# Relative errors column is calculated automatically.
def add_column(where, name, values, errors):
	err_name = "Error_%s" % name
	relerr_name = "ErrorRel_%s" % name

	where[name] = values
	where[err_name] = errors
	where[relerr_name] = where[err_name] / where[name]

	__set_formatters(where, { relerr_name: __varlist_formatters["ErrorRel"] })

# add a column by @name (str) from variables with same @name in @varlists
# (list of "varlist" objects) into a "column" object at @where
#
# As a convenience measure, @varlists can be a dict of "varlist" objects,
# but then @indices must be a list of keys to that dict.
#
# It is unspecified whether relative errors column is recalculated or inherited.
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
# use `varlist()` for an empty varlist
#
# Relative errors column is not calculated.
def varlist(*args, **kwargs):
	df = pd.DataFrame(*args, **kwargs, columns = ["Value", "Error", "ErrorRel"])
	__set_formatters(df, __varlist_formatters)
	return df

# creates a "varlist" object by variable @names (list), @values (list) and @errors (list)
#
# Relative errors column is calculated automatically.
def var_many(names, values, errors):
	return varlist({ "Value": values,
	                 "Error": errors,
	                 "ErrorRel": [e/v for e, v in zip(errors, values)] },
	               index = names)

# creates a "varlist" object by single variable @name, @value and @error
#
# Relative errors column is calculated automatically.
def var(name, value, error):
	return var_many([name], [value], [error])

# adds multiple varlists @args to a varlist @where
def add(where, *args):
	for arg in args:
		for k, v in arg.iterrows():
			where.loc[k] = v

# adds multiple varlists @args to an iterable (list) of varlists @targets
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

# reads a CSV file @name, returning a "varlist" or "columns" object as appropriate
#
# For "varlist" objects, the CSV file must have columns "Value" and "Error" and
# an unnamed first column.
#
# For "columns" objects, the CSV file must not parse as a "varlist" object
# (i. e. no "Value" and "Error" columns).
#
# In all cases, relative errors columns are calculated automatically.
def read_csv(name):
	csv = pd.read_csv(name)
	if "Value" in csv.columns and \
	   "Error" in csv.columns:
		# compute ErrorRel for a varlist
		csv["ErrorRel"] = csv["Error"] / csv["Value"]
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
			relerr_col = "ErrorRel_%s" % var_col

			# insert the new column after the error column
			csv.insert(list(csv.columns).index(err_col) + 1,
			           relerr_col,
			           csv[err_col] / csv[var_col])

			# save its name
			relerr_cols.append(relerr_col)

		# set percentage format for all these columns
		__set_formatters(csv, { k: __varlist_formatters["ErrorRel"]
		                        for k in relerr_cols })

	return csv

# reads a CSV file @name, returning an empty "varlist" object in failure case
#
# NOTE: a "varlist" object is not a "columns" object.
def maybe_read_csv(name):
	try:
		return read_csv(name)
	except OSError:
		return varlist()

# reads a directory of CSV files @name, returning a dictionary of appropriate objects
# or an empty dictionary in failure case
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

# reads a typical experiment file hierarchy at current directory
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

		print("Initial guesses for %s:\n" % name, fit_result)
		initial = beta

	# use ODR (Deming regression) which is a special case of TLS (total least squares)
	# to find results accounting for both X and Y uncertainties
	odr_model = sp_odr.Model(lambda B, x: model(x, *B))
	odr_data = sp_odr.RealData(x = x, y = y, sx = xerr, sy = yerr)
	odr = sp_odr.ODR(odr_data, odr_model, beta0 = initial, maxit = int(1e6))
	odr_output = odr.run()

	fit_result = var_many(names = model_args, values = odr_output.beta, errors = odr_output.sd_beta)
	disp("Final guesses for %s:" % name, fit_result)

	return fit_result

def fit2(name, model, x, y, xerr, yerr, data, initial = None, **kwargs):
	model_args = list(inspect.signature(model).parameters.keys())[1:]
	result = fit(name, model, model_args, x, y, xerr, yerr, initial, **kwargs)
	add(data, result)

	return lambda x: model(x, *[data.Value[a] for a in model_args])

def minmax(arg):
	return min(arg), max(arg)

def linspace(arg, ticks = 100, pre = 0, post = 0.1):
	min_arg, max_arg = minmax(arg)
	return np.linspace(min_arg - pre * (max_arg - min_arg),
	                   max_arg + post * (max_arg - min_arg),
	                   ticks)

# computes error of given expression (symbolically) given list of its variables (to consider in the calculation)
# returns:
# - error expression
# - variables representing errors of given variables
# - list of derivatives of given variables
# - list of squared error*derivative of given variables
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

# computes a substitution dictionary for the .subs() method of the symbolic expression
# takes:
# - list of variables to substitute
# - list of variables representing errors of given variables to substitute
# - a DataFrame in usual format with variables' data
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

# computes a symbolic expression along with its error from given data
# takes:
# - the expression name (for logging)
# - the expression
# - a DataFrame in usual format with variables' data
# TODO: autogenerate sympy symbols for function arguments (considering their names)
#       using some kind of introspection
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
# has column names instead of values
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
# has column names instead of values
# generates default column names by convention rather than by mapping
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

# data is a DataFrame with common constants for all instances of the computation
# cols is a DataFrame with columns
# cols_mapping is formatted as a dictionary of dict[<var>]["Error", "Value"] = <column name>
#              if None, columns are matched by names (errors as Error_<var>)
# TODO: support mixed modes where some columns are specified in cols_mapping but the remaining
#       are inferred (e. g. common error column for multiple properly named data columns)
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

def castable_to_float(arg):
	try:
		arg = float(arg)
		return True
	except:
		return False

# computes a substitution dictionary for the .subs() method of the symbolic expression
# takes:
# - list of variables to substitute
# - list of variables representing errors of given variables to substitute
# - a dictionary formatted as input to var_dict() (items which are not numbers are ignored)
def sym_make_subs_aux(expr_vars, expr_err_vars, aux):
	var_pairs = { var: aux[var.name]["Value"]
	              for var
	              in expr_vars
	              if var.name in aux
	              and "Value" in aux[var.name]
	              and castable_to_float(aux[var.name]["Value"]) }

	err_pairs = { err_var: aux[var.name]["Error"]
	              for var, err_var
	              in zip(expr_vars, expr_err_vars)
	              if var.name in aux
	              and "Error" in aux[var.name]
	              and castable_to_float(aux[var.name]["Error"]) }

	var_pairs.update(err_pairs)
	return var_pairs

def castable_to_iter(arg):
	try:
		arg = iter(arg)
		return True
	except:
		return False

# computes a substitution dictionary template for the .subs() method of the symbolic expression
# (with column references instead of values)
# takes:
# - list of variables to substitute
# - list of variables representing errors of given variables to substitute
# - a dictionary formatted as input to var_dict() (items which are not numbers are ignored)
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
# Takes:
# - name: the output variable name
# - expr: sympy expression or a function
#         NB: if expr is a function, then its argument names are significant;
#             they should match column and variable names in the dataset
# - data: a varlist DataFrame which is to be used for all instances of the computation
# - cols: a "columns" DataFrame which is to be substituted row-by-row
# - aux: a dict, formatted as input to var_dict(), which can contain either immediate values
#        (which are then substituted like values from data) or strings referring to columns
#        from cols, forming a mapping from expression's variables (and their errors) to columns
#
# Returns:
# - the sympified expression (valuable if a function is passed, but the expression is needed afterwards)
# - its variables
# - its error variables, in the same order
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

#  vim: set ts=8 sw=8 tw=0 noet ft=python :
