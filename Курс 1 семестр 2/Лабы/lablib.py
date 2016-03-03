import pandas as pd
import numpy as np
from scipy import optimize as sp_opt
from scipy import odr as sp_odr
from IPython.display import display as disp
import sympy as smp

def varlist():
	return pd.DataFrame(columns = ["Value", "Error"])

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
	return var_dict({ name: { "Value": value, "Error": error } })

def var_many(names, values, errors):
	return pd.DataFrame({ "Value": values, "Error": errors }, index = names)

def fit(name, model, model_args, x, y, xerr, yerr):
	# use OLS (ordinary least squares) to find initial guesses
	beta, cov = sp_opt.curve_fit(model,
	                             xdata = x,
	                             ydata = y,
	                             sigma = yerr,
	                             absolute_sigma = True, maxfev = 1000000)

	fit_result = var_many(names = model_args, values = beta, errors = [cov[i, i]**0.5 for i, v in enumerate(cov)])
	print("Initial guesses for %s:" % name, fit_result)

	# use ODR (Deming regression) which is a special case of TLS (total least squares)
	# to find results accounting for both X and Y uncertainties
	odr_model = sp_odr.Model(lambda B, x: model(x, *B))
	odr_data = sp_odr.RealData(x = x, y = y, sx = xerr, sy = yerr)
	odr = sp_odr.ODR(odr_data, odr_model, beta0 = beta, maxit = 1000000)
	odr_output = odr.run()

	fit_result = var_many(names = model_args, values = odr_output.beta, errors = odr_output.sd_beta)
	disp("Final guesses for %s:" % name, fit_result)

	return fit_result

def minmax(arg):
	return min(arg), max(arg)

def linspace(arg, ticks = 100):
	min_arg, max_arg = minmax(arg)
	return np.linspace(min_arg, max_arg + 0.1 * (max_arg - min_arg), ticks)

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
		err_var = smp.symbols("error_%s" % var.name)
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

# computes a symbolic expression along with its error from given data
# takes:
# - the expression name (for logging)
# - the expression
# - a DataFrame in usual format with variables' data
def sym_compute(name, expr, data):
	expr_vars = expr.atoms(smp.Symbol)
	expr_err, expr_err_vars, expr_err_derivs, expr_err_e_d_sq = sym_error(expr, expr_vars)
	expr_subs = sym_make_subs(expr_vars, expr_err_vars, data)

	bits = var_dict({ var.name: { "Error": data.Error[var.name],
	                              "Derivative": deriv.subs(expr_subs),
				      "(E*D)^2": e_d_sq.subs(expr_subs) }
	                  for var, deriv, e_d_sq in zip(expr_vars, expr_err_derivs, expr_err_e_d_sq) },
	                  index = ["Error", "Derivative", "(E*D)^2"])
	disp("Error influence estimations for %s:" % name, bits)

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

# data is a DataFrame common constants for all instances of the computation
# cols_mapping is formatted as input to `var_dict`, except with column names instead of values
# cols is a DataFrame with columns
def sym_compute_column(name, expr, data, cols_mapping, cols):
	expr_vars = expr.atoms(smp.Symbol)
	expr_err, expr_err_vars, expr_err_derivs, expr_err_e_d_sq = sym_error(expr, expr_vars)
	expr_subs = sym_make_subs(expr_vars, expr_err_vars, data)

	# pre-substitute constants
	expr = expr.subs(expr_subs)
	expr_err = expr_err.subs(expr_subs)

	expr_subs_cols_mapping = sym_make_subs_cols_mapping(expr_vars, expr_err_vars, cols_mapping)
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
#  vim: set ts=8 sw=8 tw=0 noet ft=python :
