import numpy as np
from scipy.sparse.linalg import cg as scipy_cg
import scipy.sparse as sps

import jlinops


def newton_krylov(obj_fn, grad_fn, inv_hess_fn, x0=None, backtrack_alpha=0.5, backtrack_beta=0.5, newton_tol=1e-3, cg_tol=1e-3, newton_maxiter=20, cg_maxiter=None, n=None, search_maxiter=30):
    """Performs Newton-Krylov iteration until a convergence criterion is met.

    obj_fn: function to be minimized. Should returna scalar.
    grad_fn: gradient of the obj_fn. Should return a vector.
    inv_hess_fn: inverse of hessian of the obj_fn. Should return a linear operator.
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
    
    newton_dir = None
    converged = False
    n_iters = 0

    # Perform Newton-Kyrlov iteration.
    for j in range(newton_maxiter):

        # Check whether to terminate
        if grad_norm < newton_tol:
            converged = True
            break
        
        # Compute current gradient inverse hessian.
        grad = grad_fn(x)
        grad_norm = np.linalg.norm(grad_fn(x))
        grad_norm_obj_vals.append(grad_norm)
        inv_hess_op = inv_hess_fn(x)

        # Solve with conjugate gradient method to obtain Newton direction.
        newton_dir, _ = scipy_cg(inv_hess_op, -grad, x0=newton_dir, tol=cg_tol, maxiter=cg_maxiter)

        # Perform line search to select step size
        step_size = 1.0
        searching = ( obj_val - obj_fn(x + step_size*newton_dir) ) < -backtrack_alpha*step_size*np.dot(grad, newton_dir)
        for i in range(search_maxiter):
            if not searching:
                break
            else:
                step_size *= backtrack_beta
                searching = ( obj_val - obj_fn(x + step_size*newton_dir) ) < -backtrack_alpha*step_size*np.dot(grad, newton_dir)

        # Take step
        x = x + step_size*newton_dir
        n_iters += 1

        # Evaluate objective
        obj_val = obj_fn(x)
        obj_vals.append(obj_val)
        
    print(n_iters)
    data = {
        "x": x,
        "n_iters": n_iters,
        "obj_vals": obj_vals,
        "grad_norm_obj_vals": grad_norm_obj_vals,
        "converged": converged
    }

    return data





def despeckling_newton_krylov(A, R, theta, L, f, hess_clip=1e-3, u0=None, backtrack_alpha=0.5, backtrack_beta=0.5, newton_tol=1e-3, cg_tol=1e-3, newton_maxiter=20, cg_maxiter=None, n=None, search_maxiter=30):
    """Performs a projected Newton-Krylov iteration for solving a subproblem in 
    a block coordinate descent method for despeckling.

    obj_fn: function to be minimized. Should returna scalar.
    grad_fn: gradient of the obj_fn. Should return a vector.
    inv_hess_fn: inverse of hessian of the obj_fn. Should return a linear operator.
    """

    # Build Rtilde
    Rtilde = jlinops.DiagonalOperator(1.0/np.sqrt(theta)) @ R

    # Functions
    def obj_fn(u):
        Au = A.matvec(u)
        return L*(np.log( Au ).sum() + (f/Au).sum()) + 0.5*(np.linalg.norm(Rtilde.matvec(u))**2)

    def grad_fn(u):
        Au = A.matvec(u)
        p1 = L*A.rmatvec( ( (1.0/(Au)) - (f/(Au**2)) ) )
        p2 = Rtilde.T @ (Rtilde @ u)
        
        return p1 + p2

    def inv_hess_fn(u):
        Au = A.matvec(u)
        #D1 = jlinops.DiagonalOperator( ( (2*f)/(Au**3) ) - ( 1.0/(Au**2) ) )
        D1 = jlinops.DiagonalOperator(  np.clip( ( (2*f)/(Au**3) ) - ( 1.0/(Au**2) ) , a_min=hess_clip, a_max=2*f - hess_clip) )

        p1 = L*( A.T @ ( D1 @ A ) )
        p2 = Rtilde.T @ Rtilde

        return p1 + p2
    
  
    if u0 is None:
        assert n is not None, "Must provide kwarg n if giving an initialization."
        u = np.ones(n)
    else:
        u = u0

    obj_val = obj_fn(u)
    obj_vals = [obj_val]
    grad = grad_fn(u)
    grad_norm = np.linalg.norm(grad)
    grad_norm_obj_vals = []
    
    newton_dir = None
    converged = False
    n_iters = 0

    # Perform Newton-Kyrlov iteration.
    for j in range(newton_maxiter):

        # Check whether to terminate
        if grad_norm < newton_tol:
            converged = True
            break
        
        # Compute current gradient inverse hessian.
        grad = grad_fn(u)
        grad_norm = np.linalg.norm(grad_fn(u))
        grad_norm_obj_vals.append(grad_norm)
        inv_hess_op = inv_hess_fn(u)

        # Solve with conjugate gradient method to obtain Newton direction.
        newton_dir, _ = scipy_cg(inv_hess_op, -grad, x0=newton_dir, tol=cg_tol, maxiter=cg_maxiter)
        projected_newton_dir = newton_dir


        # Perform line search to select step size
        step_size = 1.0
        searching = ( obj_val - obj_fn( u + step_size*newton_dir) )  < -backtrack_alpha*step_size*np.dot(grad, projected_newton_dir)
        for i in range(search_maxiter):
            if not searching:
                break
            else:
                step_size *= backtrack_beta
                searching = ( obj_val - obj_fn( u + step_size*newton_dir) ) < -backtrack_alpha*step_size*np.dot(grad, projected_newton_dir)

        # Take step
        u = u + step_size*newton_dir
        n_iters += 1

        # Evaluate objective
        obj_val = obj_fn(u)
        obj_vals.append(obj_val)
        
    print(n_iters)
    data = {
        "u": u,
        "n_iters": n_iters,
        "obj_vals": obj_vals,
        "grad_norm_obj_vals": grad_norm_obj_vals,
        "converged": converged
    }

    return data



def despeckling_priorconditioned_newton_krylov(f, L, A, Rtilde, Rpinv, W, u0=None, hess_clip=1e-3, backtrack_alpha=0.5, backtrack_beta=0.5, newton_tol=1e-3, newton_maxiter=20, n=None, search_maxiter=30, **kwargs):
    """Implements a priorconditioned Newton-Krylov method for solving a subproblem of the despeckling coordinate descent method.

    Rpinv: a linear operator representing the pseudoinverse of Rtilde.
    W: a matrix s.t. col(W) = ker(R).
    """

    # Functions
    def obj_fn(u):
        Au = A.matvec(u)
        return L*(np.log( Au ).sum() + (f/Au).sum()) + 0.5*(np.linalg.norm(Rtilde.matvec(u))**2)

    def grad_fn(u):
        Au = A.matvec(u)
        p1 = L*A.rmatvec( ( (1.0/(Au)) - (f/(Au**2)) ) )
        p2 = Rtilde.T @ (Rtilde @ u)
        
        return p1 + p2

    def inv_hess_fn(u):
        Au = A.matvec(u)
        D1 = jlinops.DiagonalOperator(  np.clip( ( (2*f)/(Au**3) ) - ( 1.0/(Au**2) ) , a_min=hess_clip, a_max=2*f - hess_clip) )

        p1 = L*( A.T @ ( D1 @ A ) )
        p2 = Rtilde.T @ Rtilde

        return p1 + p2
    

    if u0 is None:
        assert n is not None, "Must provide kwarg n if giving an initialization."
        u = np.ones(n)
    else:
        u = u0
        

    obj_val = obj_fn(u)
    obj_vals = [obj_val]
    grad = grad_fn(u)
    grad_norm = np.linalg.norm(grad)
    grad_norm_obj_vals = []
    
    newton_dir = None
    last_cgls_sol = None
    converged = False
    n_iters = 0

    # Perform Newton-Kyrlov iteration.
    for j in range(newton_maxiter):

        # Check whether to terminate
        if grad_norm < newton_tol:
            converged = True
            break
        
        # Compute current gradient inverse hessian.
        grad = grad_fn(u)
        grad_norm = np.linalg.norm(grad_fn(u))
        grad_norm_obj_vals.append(grad_norm)


        # Do a bunch of stuff to set up the solve for the Newton direction
        Au = A.matvec(u)
        s1 = L*( (1.0/Au) - (f/(Au**2))  )
        s2 = L*( 2.0*(f/(Au**3)) - (1.0/(Au**2)) )
        s2 = np.clip(s2, a_min=hess_clip, a_max=2*f - hess_clip)

        y1 = (1.0/np.sqrt(s2))*s1
        y2 = Rtilde.matvec(u)

        B = jlinops.MatrixLinearOperator(sps.diags(np.sqrt(s2))) @ A
        if W is not None:
            BWpinv = jlinops.QRPinvOperator( jlinops.MatrixLinearOperator( B @ W.A ) )
        else:
            BWpinv = 0.0

        # Solve for the Newton direction with priorconditioned Kyrlov method
        cgls_solve = jlinops.trlstsq_standard_form(B, -y1, Rpinv=Rpinv, R=Rtilde,
                                                    AWpinv=BWpinv, lam=1.0, shift=-y2, W=W, 
                                                    initialization=last_cgls_sol, **kwargs)
        newton_dir = cgls_solve["x"]
        last_cgls_sol = newton_dir

        # Perform line search to select step size
        step_size = 1.0
        searching = ( obj_val - obj_fn( u + step_size*newton_dir) )  < -backtrack_alpha*step_size*np.dot(grad, newton_dir)
        for i in range(search_maxiter):
            if not searching:
                break
            else:
                step_size *= backtrack_beta
                searching = ( obj_val - obj_fn( u + step_size*newton_dir) ) < -backtrack_alpha*step_size*np.dot(grad, newton_dir)

        # Take step
        u = u + step_size*newton_dir
        n_iters += 1

        # Evaluate objective
        obj_val = obj_fn(u)
        obj_vals.append(obj_val)
        
    print(n_iters)
    data = {
        "u": u,
        "n_iters": n_iters,
        "obj_vals": obj_vals,
        "grad_norm_obj_vals": grad_norm_obj_vals,
        "converged": converged
    }

    return data





def log_despeckling_priorconditioned_newton_krylov(log_data, L, Rtilde, Rpinv, W, x0=None, backtrack_alpha=0.5, backtrack_beta=0.5, newton_tol=1e-3, newton_maxiter=20, n=None, search_maxiter=30, **kwargs):
    """Implements a priorconditioned Newton-Krylov method for solving a subproblem of the
    log-domain despeckling coordinate descent method.

    Rpinv: a linear operator representing the pseudoinverse of Rtilde.
    W: a matrix s.t. col(W) = ker(R).
    """

    obj_fn = lambda x: L*( np.exp(log_data - x) + x - log_data ).sum() + 0.5*((np.linalg.norm( Rtilde @ x ))**2)
    grad_fn = lambda x: L*(1 - np.exp(log_data - x)) + Rtilde.rmatvec(Rtilde.matvec(x))

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
    
    newton_dir = None
    last_cgls_sol = None
    converged = False
    A = jlinops.IdentityOperator((len(x), len(x)))
    n_iters = 0

    # Perform Newton-Kyrlov iteration.
    for j in range(newton_maxiter):

        # Check whether to terminate
        #print(grad_norm)
        if grad_norm < newton_tol:
            converged = True
            break

        # Compute current gradient and norm
        grad = grad_fn(x)
        grad_norm = np.linalg.norm(grad)
        grad_norm_obj_vals.append(grad_norm)
        
        # Do a bunch of stuff to set up the solve for the Newton direction
        s1 = L*(1.0 - np.exp(log_data - x))
        s2 = L*np.exp(log_data - x)

        y1 = (1.0/np.sqrt(s2))*s1
        y2 = Rtilde.matvec(x)

        B = jlinops.MatrixLinearOperator(sps.diags(np.sqrt(s2))) @ A
        BWpinv = jlinops.QRPinvOperator( jlinops.MatrixLinearOperator( B @ W.A ) )

        # Solve for the Newton direction with priorconditioned Kyrlov method
        cgls_solve = jlinops.trlstsq_standard_form(B, -y1, Rpinv=Rpinv, R=Rtilde,
                                                    AWpinv=BWpinv, lam=1.0, shift=-y2, W=W, 
                                                    initialization=last_cgls_sol, **kwargs)
        newton_dir = cgls_solve["x"]
        last_cgls_sol = newton_dir


        # Perform line search to select step size
        step_size = 1.0
        searching = ( obj_val - obj_fn(x + step_size*newton_dir) ) < -backtrack_alpha*step_size*np.dot(grad, newton_dir)
        for i in range(search_maxiter):
            if not searching:
                break
            else:
                step_size *= backtrack_beta
                searching = ( obj_val - obj_fn(x + step_size*newton_dir) ) < -backtrack_alpha*step_size*np.dot(grad, newton_dir)

        # Take step
        x = x + step_size*newton_dir
        n_iters += 1

        # Evaluate objective
        obj_val = obj_fn(x)
        obj_vals.append(obj_val)

    print(n_iters)
    data = {
        "x": x,
        "n_iters": n_iters,
        "obj_vals": obj_vals,
        "grad_norm_obj_vals": grad_norm_obj_vals,
        "converged": converged
    }

    return data
    
    




