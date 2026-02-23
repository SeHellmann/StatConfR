#ifndef NELDER_MEAD_H
#define NELDER_MEAD_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>
#include <cfloat>

// Thread-safe, standalone Nelder-Mead implementation.

typedef double (*nm_objective_fn)(int n, double *par, void *ex);

inline void nelder_mead_optimize(int n, double *start_par, double *final_par, double *final_val,
                                 nm_objective_fn fn, int *fail, 
                                 double tol, void *ex,
                                 int *fncount, int maxit) {
    
    const double alpha = 1.0;
    const double beta = 0.5;
    const double gamma = 2.0; 
    const double delta = 0.5;

    std::vector<std::vector<double>> simplex(n + 1, std::vector<double>(n));
    std::vector<double> values(n + 1);
    
    for(int j=0; j<n; ++j) simplex[0][j] = start_par[j];
    values[0] = fn(n, simplex[0].data(), ex);
    (*fncount)++;
    
    for(int i=1; i<=n; ++i) {
        for(int j=0; j<n; ++j) simplex[i][j] = start_par[j];
        
        if (std::abs(simplex[i][i-1]) > 1e-4) {
            simplex[i][i-1] *= 1.1;
        } else {
            simplex[i][i-1] = 0.001;
        }
        
        values[i] = fn(n, simplex[i].data(), ex);
        (*fncount)++;
    }
    
    int iter = 0;
    *fail = 1;
    
    std::vector<int> indices(n + 1);
    for(int i=0; i<=n; ++i) indices[i] = i;
    
    std::vector<double> centroid(n);
    std::vector<double> xr(n);
    std::vector<double> xe(n);
    std::vector<double> xc(n);
    
    while(iter < maxit) {
        iter++;
        
        std::sort(indices.begin(), indices.end(), [&](int a, int b){
            return values[a] < values[b];
        });
        
        double best_val = values[indices[0]];
        double worst_val = values[indices[n]];
        
        if (iter > 1) {
             if ((worst_val - best_val) < tol * (std::abs(best_val) + tol)) {
                 *fail = 0;
                 break;
             }
        }
        
        for(int j=0; j<n; ++j) centroid[j] = 0.0;
        int worst_idx = indices[n];
        
        for(int i=0; i<n; ++i) {
            int idx = indices[i];
            for(int j=0; j<n; ++j) {
                centroid[j] += simplex[idx][j];
            }
        }
        for(int j=0; j<n; ++j) centroid[j] /= n;
        
        for(int j=0; j<n; ++j) xr[j] = centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j]);
        double vr = fn(n, xr.data(), ex);
        (*fncount)++;
        
        if (vr < values[indices[n-1]] && vr >= values[indices[0]]) {
            simplex[worst_idx] = xr;
            values[worst_idx] = vr;
            continue;
        }
        
        if (vr < values[indices[0]]) {
            for(int j=0; j<n; ++j) xe[j] = centroid[j] + gamma * (xr[j] - centroid[j]);
            double ve = fn(n, xe.data(), ex);
            (*fncount)++;
            
            if (ve < vr) {
                simplex[worst_idx] = xe;
                values[worst_idx] = ve;
            } else {
                simplex[worst_idx] = xr;
                values[worst_idx] = vr;
            }
            continue;
        }
        
        bool contraction_accepted = false;
        
        if (vr < values[worst_idx]) {
            for(int j=0; j<n; ++j) xc[j] = centroid[j] + beta * (xr[j] - centroid[j]);
            double vc = fn(n, xc.data(), ex);
            (*fncount)++;
            
            if (vc <= vr) {
                simplex[worst_idx] = xc;
                values[worst_idx] = vc;
                contraction_accepted = true;
            }
        } else {
            for(int j=0; j<n; ++j) xc[j] = centroid[j] - beta * (xr[j] - centroid[j]);
             for(int j=0; j<n; ++j) xc[j] = centroid[j] + beta * (simplex[worst_idx][j] - centroid[j]);
             
             double vc = fn(n, xc.data(), ex);
             (*fncount)++;
             
             if (vc < values[worst_idx]) {
                 simplex[worst_idx] = xc;
                 values[worst_idx] = vc;
                 contraction_accepted = true;
             }
        }
        
        if (contraction_accepted) continue;
        
        int best_idx = indices[0];
        for(int i=1; i<=n; ++i) {
            int idx = indices[i];
            for(int j=0; j<n; ++j) {
                simplex[idx][j] = simplex[best_idx][j] + delta * (simplex[idx][j] - simplex[best_idx][j]);
            }
            values[idx] = fn(n, simplex[idx].data(), ex);
            (*fncount)++;
        }
    }
    
    int best_idx = indices[0];
    for(int j=0; j<n; ++j) final_par[j] = simplex[best_idx][j];
    *final_val = values[best_idx];
}

#endif
