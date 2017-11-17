#include <stdlib.h>

void calculate_covariances9(
    double *x,
    double *v,
    double *ums,
    double *covariances,
    int n, int c, int d
){
    double *xv = (double*)malloc(d*sizeof(*xv));
    
    for (int i = 0; i < c; i++){
        double sum_u = 0.0;
        
        for (int k = 0; k < n; k++){
            double um = ums[i*n + k];
            sum_u += um;
            
            for (int p = 0; p < d; p++){
                xv[p] = x[k*d + p] - v[i*d + p];
            }
            
            for (int p = 0; p < d; p++){
                for (int q = 0; q < d; q++){
                    covariances[i*d*d + p*d + q] += um * xv[p] * xv[q];
                }
            }
        }
        
        for (int p = 0; p < d; p++){
                for (int q = 0; q < d; q++){
                    covariances[i*d*d + p*d + q] /= sum_u;
                }
            }
        }

    free(xv);
}
