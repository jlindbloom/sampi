import numpy as np



def gradient_descent(obj_fn, grad_fn, x0=None, backtrack_alpha=0.5, backtrack_beta=0.5, tol=1e-3, maxiter=30, n=None, search_maxiter=30):
    """Performs gradient descent iteration until a convergence criterion is met.

    obj_fn: function to be minimized. Should returna scalar.
    grad_fn: gradient of the obj_fn. Should return a vector.
    """

    if x0 is None:
        assert n is not None, "Must provide kwarg n if giving an initialization."
        x = np.ones(n)
    else:
        x = x0

    obj_val = obj_fn(x)
    obj_vals = [obj_val]
    grad = grad_fn(x)
    grad_norm = np.linalg.norm(grad)
    grad_norm_obj_vals = []

    converged = False
    n_iters = 0

    # Perform gradient descent
    for j in range(maxiter):

         # Check whether to terminate
        if grad_norm < tol:
            converged = True
            break

        # Compute current gradient inverse hessian.
        grad = grad_fn(x)
        grad_norm = np.linalg.norm(grad_fn(x))
        grad_norm_obj_vals.append(grad_norm)

        # Gradient
        grad = grad_fn(x)

        # Line search
        gfunc = lambda t: obj_fn(x - t*grad)
        step_size = 1.0
        searching = ( obj_val - gfunc(step_size) ) < backtrack_alpha*step_size*(grad_norm**2)
        for i in range(search_maxiter):
            if not searching:
                break
            else:
                step_size *= backtrack_beta
                searching = ( obj_val - gfunc(step_size) ) < backtrack_alpha*step_size*(grad_norm**2)
        
        # Take step
        x = x - step_size*grad
        n_iters += 1

        # Evaluate objective
        obj_val = obj_fn(x)
        obj_vals.append(obj_val)
        
    data = {
        "x": x,
        "n_iters": n_iters,
        "obj_vals": obj_vals,
        "grad_norm_obj_vals": grad_norm_obj_vals,
        "converged": converged
    }

    return data

















