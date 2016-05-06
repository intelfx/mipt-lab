import pandas as pd
import numpy as np
from scipy import optimize as sp_opt
from scipy import odr as sp_odr
from IPython.display import display as disp
import sympy as smp
import inspect
import os

try:
	from natsort import natsorted
except ImportError:
	pass

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

def varlist():
	return pd.DataFrame(columns = ["Value", "Error", "ErrorRel"])

def add(where, *args):
	for arg in args:
		for k, v in arg.iterrows():
			where.loc[k] = v

def add_multi(targets, *args):
	for where in targets:
		add(where, *args)

def read_csv(name):
	return pd.read_csv(name)

def var_dict(dct, **kwargs):
	return pd.DataFrame(dct, **kwargs).T

def var(name, value, error):
	return var_dict({ name: { "Value": value, "Error": error, "ErrorRel": error / value } })

def var_many(names, values, errors):
	return pd.DataFrame({ "Value": values,
	                      "Error": errors,
	                      "ErrorRel": [e/v for e, v in zip(errors, values)] },
	                    columns = ["Value", "Error", "ErrorRel"],
	                    index = names)

def read_standard_layout():
	data = {}
	columns = {}

	try:
		constants = read_csv("constants.csv")
	except OSError:
		pass

	for f in os.listdir("constants"):
		if f.endswith(".csv"):
			e = f[:-4]
			data[e] = read_csv(os.path.join("constants", f))

	for f in os.listdir("measurements"):
		if f.endswith(".csv"):
			e = f[:-4]
			columns[e] = read_csv(os.path.join("measurements", f))

	for e in columns.keys():
		d = varlist()
		if 'constants' in locals():
			add(d, constants)
		if e in data:
			add(d, data[e])
		data[e] = d

	# also add "global" varlist with only the global constants
	d = varlist()
	if 'constants' in locals():
		add(d, constants)
	data["global"] = d

	if 'natsorted' in globals():
		experiments = natsorted(columns.keys())
	else:
		experiments = sorted(columns.keys())

	return data, columns, experiments

def fit(name, model, model_args, x, y, xerr, yerr, initial = None):
	# use OLS (ordinary least squares) to find initial guesses
	if initial is None:
		beta, cov = sp_opt.curve_fit(model,
		                             xdata = x,
		                             ydata = y,
		                             sigma = yerr,
		                             absolute_sigma = True, maxfev = int(1e6))

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

def fit2(name, model, x, y, xerr, yerr, data, initial = None):
	model_args = list(inspect.signature(model).parameters.keys())[1:]
	result = fit(name, model, model_args, x, y, xerr, yerr, initial)
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
	bits = var_dict({ var.name: { "Error": data.Error[var.name] if var.name in data.Error else None,
	                              "Derivative": deriv.subs(expr_subs),
	                              "(E*D)^2": e_d_sq.subs(expr_subs) }
	                  for var, deriv, e_d_sq in zip(expr_vars, expr_err_derivs, expr_err_e_d_sq) },
	                index = ["Error", "Derivative", "(E*D)^2"])

	bits = bits.sort_values("(E*D)^2", ascending=False)

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
# cols_mapping is formatted as input to `var_dict`, except with column names instead of values
#              if None, columns are matched by names (errors as Error_<var>)
# TODO: support mixed modes where some columns are specified in cols_mapping but the remaining
#       are inferred (e. g. common error column for multiple properly named data columns)
def sym_compute_column(name, expr, data, cols, cols_mapping = None):
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
def compute(name, expr, data, columns = None, aux = None, debug = False):
	if type(expr).__name__ == "function":
		expr_fn = expr
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

		columns[name] = expr_rows
		columns["Error_%s" % name] = expr_err_rows
	else:
		expr_out = var(name, float(expr), float(expr_err))

		if debug:
			print("Result:")
			disp(expr_out)

		add(data, expr_out)

	return expr, expr_vars, expr_err_vars

#  vim: set ts=8 sw=8 tw=0 noet ft=python :
