/* Program file RLspectra.c - 
* Processes all generation of spectra
*
* Last change: SJ 22.03.24
*
============================================
*
* RLexact: The exact diagonalization package
* Christian Rischel & Kim Lefmann, 26.02.94  
* Version 4.0, September 2017 HVILKEN VERSION? :D
* 
============================================
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <glob.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <math.h>

#define RETRY 3
#define DIMENSION 2
#define PDIMENSION 2
#define DEFAULT_PRECISION 1e-8
#define MASS 1
#define STEP_SIZE 0.1
#define NUM_STEPS 100
#define NUM_SAMPLES 1000

const char *pattern = "./12x1x1_antiSim.f2.m0.szz";
double param[2] = {0.1, 0.1};
double minLogVal = log(DBL_MIN * 100);
bool train = false;
bool trainedBefore = false;
bool testOutput = false;
bool firstTime = true;
bool predict = true;
double noise = 1e-4;
double singleCoords[2] = {0.5, 5};
double predMin_k = 0;
double predMax_k = 1;
double predMin_E = 0;
double predMax_E = 2*M_PI;
int nk = 100;
int nE = 100;
int predInt = 2;
int predLen = 1;
const char *predFile = "PredictionResults.txt";


/* Functions declared in this file */
void print_matrix(gsl_matrix *m);
void read_matrix(const char *filename);
int readFile(const char *filename, double **qArray, double **eArray, double **szzArray);
int multiReadFile(const char **filenames, size_t nFiles, double **qArray, double **eArray, double **szzArray);
double sqrCovFunc(double *x1, double *x2, double sigma, double l);
void cov_matrix(gsl_matrix* result, gsl_matrix* x1, gsl_matrix* x2, double* parameters);
void write_matrix(gsl_matrix *m, const char *filename, char *header);
void inversion_error_handler(const char * reason, const char * file, int line, int gsl_errno);
int find_index(double *array, int size, double value);
double normalizationCoef(gsl_matrix *prediction, int N, double dk, double dE);
void advanced_Gauss_predict(gsl_matrix** prediction, gsl_matrix* pred_coords, gsl_matrix* data_values, gsl_matrix* data_coords, double* parameters);
double hmc_sample(double **parameters, gsl_matrix *x_values, gsl_matrix *data_values, double noise, gsl_rng *rng);
void leapfrog(double (*parameters)[DIMENSION], double (*momentum)[DIMENSION], double *loglikelihood, gsl_matrix **gradient, gsl_matrix **pureCovMat, gsl_matrix **initInvMat, gsl_matrix *noiseMat, gsl_matrix *x_values, gsl_matrix *data_values);
void GPR_predict(gsl_matrix** prediction, gsl_matrix **postCovMat, gsl_matrix* pred_coords, gsl_matrix* data_values, gsl_matrix* data_coords, double* parameters, double noise);
void GPR_single_predict(gsl_matrix** prediction, gsl_matrix **post_cov_mat, gsl_matrix **inv_init_cov, gsl_matrix* pred_coords, gsl_matrix* data_values, gsl_matrix* data_coords, double* parameters, double noise, bool *first_time);
void Gauss_predict(gsl_matrix** prediction, gsl_matrix* pred_coords, gsl_matrix* data_values, gsl_matrix* data_coords, double* parameters)
void GPR_loglikelihoodGrad(gsl_matrix **gradient, gsl_matrix *pureCovMat, gsl_matrix *initInvMat, gsl_matrix *x_values, gsl_matrix *data_values, double *parameters);
double GPR_logLikelihood(gsl_matrix* covMat, gsl_matrix* invMat, gsl_matrix* data_values);
void GPR_single_initialize(gsl_matrix** inv_init_cov, gsl_matrix* noiseMat, gsl_matrix* data_coords, double* parameters);
void writePredFile(const char *predFilename, gsl_matrix *coords, gsl_matrix *values, gsl_matrix *covariance);
void kernelDeriv(gsl_matrix **ds, gsl_matrix **dl, gsl_matrix *pureCovMat, gsl_matrix *x_values, double *parameters);
double matrix_trace(gsl_matrix *m);
double kinetic_energy(double *p);
double gaussianFunc(double distance_squared, double sigma);
double mean(double* array, int length);
void find_min_max(double* array, int length, double* min, double* max);
double matDet(gsl_matrix* m);
int invert_matrix(gsl_matrix *m, gsl_matrix **inverse);
int test_inverse(gsl_matrix *original, gsl_matrix *inverse, double precision);


/*IMPLEMENTATION*/
/**
* @brief: A custom error handler for if the sample runs into a singular matrix. Just prints the error and continues

*/
void inversion_error_handler(const char * reason, const char * file, int line, int gsl_errno)
{
    printf("GSL error: %s\n", reason);
    printf("Trying again\n");
}

double mean(double* array, int length) {
    double sum = 0.0;
    for(int i = 0; i < length; i++) {
        sum += array[i];
    }
    return sum / length;
}


void find_min_max(double* array, int length, double* min, double* max)
{
    *min = array[0];
    *max = array[0];
    for(int i = 1; i < length; i++) {
        if (array[i] < *min) {
            *min = array[i];
        }
        if (array[i] > *max) {
            *max = array[i];
        }
    }
}

int find_index(double *array, int size, double value) {
    for (int i = 0; i < size; i++) {
        if (array[i] == value) {
            return i;
        }
    }
    return -1;  // Return -1 if the value was not found
}

/**
* @brief This function prints all elements in a gsl_matrix in the format of a matrix in the terminal.
*
* @param m The matrix that will be printed.
*/
void print_matrix(gsl_matrix *m)
{
    size_t i, j;

    for (i = 0; i < m->size1; i++) {
        for (j = 0; j < m->size2; j++) {
            printf("%g ", gsl_matrix_get(m, i, j));
        }
        printf("\n");
    }
}

/**
 * Calculates the determinant of a matrix.
 *
 * @brief This function takes a gsl_matrix pointer as input and calculates the determinant of the matrix.
 *
 * @param m The gsl_matrix pointer representing the matrix.
 * @return The determinant of the matrix.
 */
double matDet(gsl_matrix* m)
{
    gsl_permutation *p = gsl_permutation_alloc(m->size1);
    int signum;
    gsl_linalg_LU_decomp(m, p, &signum);
    gsl_permutation_free(p);

    return gsl_linalg_LU_det(m, signum);
}

/**
 * Tests whether one matrix is the inverse of the other
 *
 * @brief This function takes two matrices and tests whether the product is reasonably close to the identity matrix
 *
 * @param original The original matrix
 * @param inverse The supposed inverse
 *
 * @return int Returns 1 if it is the inverse within the precision, and 0 if not
 */
int test_inverse(gsl_matrix *original, gsl_matrix *inverse, double precision)
{
    gsl_matrix *prod = gsl_matrix_alloc(original->size1, original->size2);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, original, inverse, 0.0, prod);
    for (size_t i = 0; i < original->size1; i++)
    {
        for (size_t j = 0; j < original->size2; j++)
        {
            if (i == j)
            {
                if (!(abs(gsl_matrix_get(prod, i, j)) < 1 + precision))
                {
                    printf("Precision issues on diag: size = %f\n", abs(gsl_matrix_get(prod, i, j)));
                    return 0;
                }
            }

            else
            {
                if (!abs(gsl_matrix_get(prod, i, j) < precision))
                {
                    printf("Precision issues off diag: size = %f\n", abs(gsl_matrix_get(prod, i, j)));
                    return 0;
                }
            }
        }
    }
    return 1;


    gsl_matrix_free(prod);
}

/**
 * Inverts a given matrix.
 *
 * @brief This function takes a GSL matrix `m` and calculates its inverse. The resulting
    inverse matrix is stored in the `inverse` parameter.
 *
 * @param m The matrix to be inverted.
 * @param inverse A pointer to a pointer to store the resulting inverse matrix.
 */
int invert_matrix(gsl_matrix *m, gsl_matrix **inverse)
{
    gsl_error_handler_t *old_handler = gsl_set_error_handler(inversion_error_handler);
    int s;
    size_t n = m->size1;
    gsl_matrix *copy = gsl_matrix_alloc(n, n);
    gsl_permutation *p = gsl_permutation_alloc(n);

    gsl_matrix_memcpy(copy, m);
    gsl_linalg_LU_decomp(copy, p, &s);
    int status = gsl_linalg_LU_invert(copy, p, *inverse);
    
    gsl_permutation_free(p);
    gsl_matrix_free(copy);

    gsl_set_error_handler(old_handler);
    return status == GSL_SUCCESS ? 1 : 0;
}

/**
 * Calculates the trace of a matrix.
 *
 * @brief This function takes a gsl_matrix pointer as input and calculates the trace of the matrix.
    The trace of a matrix is the sum of the elements on its main diagonal.
 *
 * @param m A pointer to the gsl_matrix.
 * @return The trace of the matrix.
 */
double matrix_trace(gsl_matrix *m)
{
    double trace = 0.0;
    size_t n = GSL_MIN(m->size1, m->size2);
    for (size_t i = 0; i < n; i++) {
        trace += gsl_matrix_get(m, i, i);
    }
    return trace;
}



/**
 * Calculates the squared covariance function between two points.
 *
 * @brief This function calculates the squared covariance function between two points using the given parameters: x1, x2, sigma, and l.
  The squared covariance function is calculated as sigma^2 * exp(-(x1-x2)^2 / (2*l^2)).
 *
 * @param x1 The first point in (hw, Szz)-space.
 * @param x2 The second point in (hw, Szz)-space.
 * @param sigma The standard deviation.
 * @param l The length scale.
 * @return double Returns the squared covariance between x1 and x2.
 */
double sqrCovFunc(double *x1, double *x2, double sigma, double l)
{
    double sum = 0;
    for (int i = 0; i < 2; i++) {
        sum += pow(x1[i] - x2[i], 2);
    }
    return sigma * sigma * exp(-sum / (2 * l * l));
}

/**
 * @brief This function calculates the covariance matrix of two vectors.
 *
 * @param result: A pointer to a gsl_matrix of shape MxN where the result will be stored.
 * @param x1: A pointer to the first gsl_matrix of shape Mx2.
 * @param x2: A pointer to the second gsl_matrix of shape Nx2.
 * @param parameters: A pointer to an array of parameters to be passed to the covariance function.
 * @param cov_function: A function pointer to the covariance function. This function should take four doubles and return a double.
 *
 * The function works by iterating over each element in the two input vectors. For each pair of elements, it calculates the covariance using the provided covariance function and parameters, and then sets the corresponding element in the result matrix to this value.
 */
void cov_matrix(gsl_matrix* result, gsl_matrix* x1, gsl_matrix* x2, double* parameters)
{
    size_t i, j;
    for(i = 0; i < x1->size1; i++)
    {
        for(j = 0; j < x2->size1; j++) 
        {
            double coords1[2] = {gsl_matrix_get(x1, i, 0), gsl_matrix_get(x1, i, 1)};
            double coords2[2] = {gsl_matrix_get(x2, j, 0), gsl_matrix_get(x2, j, 1)};
            double val = sqrCovFunc(coords1, coords2, parameters[0], parameters[1]);
            gsl_matrix_set(result, i, j, val);
        }
    }
}

/**
* @brief Updates the derivative matrices relevant for the gradient
* 
* @param ds: A pointer to a pointer to a gsl_matrix containing the entrances of the derivative with regards to σ
* @param dl: A pointer to a pointer to a gsl_matrix containing the entrances of the derivative with regards to l
* @param pureCovMat: A pointer to a gsl_matrix containing the pure noise-free covariance matrix
* @param x_values: A pointer to a gsl_matrix containing the coordinate values
* @param parameters: An array containing the Gaussian process parameters
*
*/
void kernelDeriv(gsl_matrix **ds, gsl_matrix **dl, gsl_matrix *pureCovMat, gsl_matrix *x_values, double *parameters)
{
    gsl_matrix *XX = gsl_matrix_alloc(x_values->size1, x_values->size1);
    double q1;
    double e1;
    double q2;
    double e2;
    
    gsl_matrix_memcpy(*ds, pureCovMat);
    gsl_matrix_scale(*ds, 2/parameters[0]);

    for (size_t i = 0; i < x_values->size1; i++)
    {
        for (size_t j = 0; j < x_values->size1; j++)
        {
            q1 = gsl_matrix_get(x_values, i, 0);
            e1 = gsl_matrix_get(x_values, i, 1);
            q2 = gsl_matrix_get(x_values, j, 0);
            e2 = gsl_matrix_get(x_values, j, 1);
            gsl_matrix_set(XX, i, j, pow(q1-q2, 2) + pow(e1-e2, 2));
        }
    }
    
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, XX, pureCovMat, 0.0, *dl);
    gsl_matrix_scale(*dl, pow(parameters[1], -3));
    
    gsl_matrix_free(XX);
}

/**
* @brief Updates the gradient in parameter space
* 
* @param gradient: A pointer to a pointer to a gsl_matrix containing the gradient in parameter space
* @param pureCovMat: A pointer to a gsl_matrix containing the pure noise-free covariance matrix
* @param initInvMat: A pointer to a gsl_matrix containing the inverse of the covariance matrix with noise
* @param x_values: A pointer to a gsl_matrix containing the coordinate values
* @param data_values: A pointer to a gsl_matrix containing the corresponding Szz values
* @param parameters: An array containing the Gaussian process parameters
*
*/
void GPR_loglikelihoodGrad(gsl_matrix **gradient, gsl_matrix *pureCovMat, gsl_matrix *initInvMat, gsl_matrix *x_values, gsl_matrix *data_values, double *parameters)
{
    gsl_matrix *ds = gsl_matrix_alloc(x_values->size1, x_values->size1);
    gsl_matrix *dl = gsl_matrix_alloc(x_values->size1, x_values->size1);
    gsl_matrix *alpha = gsl_matrix_alloc(x_values->size1, 1);
    gsl_matrix *gradMatrix = gsl_matrix_alloc(x_values->size1, x_values->size1);
    gsl_matrix *gradMat1 = gsl_matrix_alloc(x_values->size1, x_values->size1);
    gsl_matrix *gradMat2 = gsl_matrix_alloc(x_values->size1, x_values->size1);

    kernelDeriv(&ds, &dl, pureCovMat, x_values, parameters);

    gsl_matrix_memcpy(gradMatrix, initInvMat); //Initializes gradMatrix as invMat
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, gradMatrix, data_values, 0, alpha); //Initializes alpha as invMat@data_values
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, alpha, alpha, -1.0, gradMatrix); //Changes the sign of gradMatrix and adds alpha@alpha.T
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, gradMatrix, ds, 0.0, gradMat1); //Finds the matrix related to the σ-derivative
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, gradMatrix, dl, 0.0, gradMat2); //Finds the matrix related to the l-derivative

    gsl_matrix_set(*gradient,0, 0, 0.5 * matrix_trace(gradMat1)); //Sets the first entrance of the gradient as half the trace of gradMat1
    gsl_matrix_set(*gradient,1, 0, 0.5 * matrix_trace(gradMat2)); //Sets the second entrance of the gradient as half the trace of gradMat2

    gsl_matrix_free(ds);
    gsl_matrix_free(dl);
    gsl_matrix_free(alpha);
    gsl_matrix_free(gradMatrix);
    gsl_matrix_free(gradMat1);
    gsl_matrix_free(gradMat2);
}

/**
* @brief Calculates the log-likelihood of the Gaussian process regression
* 
* @param covMat: A pointer to a gsl_matrix containing the covariance matrix with noise
* @param invMat: A pointer to a gsl_matrix containing the inverse of the covariance matrix with noise
* @param data_values: A pointer to a gsl_matrix containing the corresponding Szz values
*
* @return: double Returns the log-likelihood
*/
double GPR_logLikelihood(gsl_matrix* covMat, gsl_matrix* invMat, gsl_matrix* data_values)
{
    double output = 0;
    gsl_matrix *tempVec = gsl_matrix_alloc(data_values->size1, 1);
    gsl_matrix *transData = gsl_matrix_alloc(1, data_values->size1);
    gsl_matrix *result = gsl_matrix_alloc(1, 1);
    
    gsl_matrix_transpose_memcpy(transData, data_values);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, invMat, data_values, 0.0, tempVec);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, transData, tempVec, 0.0, result);

    double loglikelihood = -0.5*(gsl_matrix_get(result, 0, 0) + log(matDet(covMat)) + (data_values->size1) * log(2 * M_PI));

    gsl_matrix_free(tempVec);
    gsl_matrix_free(transData);
    gsl_matrix_free(result);
    
    return loglikelihood;
}

/**
* @brief Calculates the kinetic energy for the HMC sampler
* 
* @param p: An array containing the parameter space momentum
*
* @return: double Returns the kinetic energy
*/
double kinetic_energy(double *p)
{
    double ke = 0.0;
    for (int i = 0; i < DIMENSION; i++) {
        ke += 0.5 * p[i]*p[i]/MASS;
    }
    return ke;
}

/**
* @brief Updates the log-likelihood, the gradient, the pure covariance matrix and the inverse matrix after a leapfrog integration step
* 
* @param parameters: An array containing the Gaussian process parameters
* @param momentum: An array containing the parameter space momentum
* @param loglikelihood: A pointer to the log-likelihood
* @param gradient: A pointer to a pointer to a gsl_matrix containing the gradient in parameter space
* @param pureCovMat: A pointer to a pointer to a gsl_matrix containing the pure noise-free covariance matrix
* @param initInvMat: A pointer to a pointer to a gsl_matrix containing the inverse of covariance matrix with noise
* @param noiseMat: A pointer to a gsl_matrix containing the noise
* @param x_values: A pointer to a gsl_matrix containing the coordinate values
* @param data_values: A pointer to a gsl_matrix containing the corresponding Szz values
*
*/
void leapfrog(double (*parameters)[DIMENSION], double (*momentum)[DIMENSION], double *loglikelihood, gsl_matrix **gradient, gsl_matrix **pureCovMat, gsl_matrix **initInvMat, gsl_matrix *noiseMat, gsl_matrix *x_values, gsl_matrix *data_values)
{
    gsl_matrix *sumMat = gsl_matrix_alloc((*pureCovMat)->size1, (*pureCovMat)->size2);
    int invertSuccess;


    // Full steps for position and momentum
    for (int i = 0; i < DIMENSION; i++) {
        *(parameters[i]) += STEP_SIZE * (*momentum[i]) / MASS;
    }

    
    //Updating all of the matrices
    cov_matrix(*pureCovMat, x_values, x_values, *parameters);
    gsl_matrix_memcpy(sumMat, *pureCovMat);
    gsl_matrix_add(sumMat, noiseMat);
    invertSuccess = invert_matrix(sumMat, initInvMat);
    *loglikelihood = GPR_logLikelihood(sumMat, *initInvMat, data_values);
    GPR_loglikelihoodGrad(gradient, *pureCovMat, *initInvMat, x_values, data_values, *parameters);
    }
    gsl_matrix_free(sumMat);
}

/**
* @brief Uses HMC to update the parameters in search of a likely set of parameters
* 
* @param parameters: A pointer to an array containing the Gaussian process parameters
* @param x_values: A pointer to a gsl_matrix containing the coordinate values
* @param data_values: A pointer to a gsl_matrix containing the corresponding Szz values
* @param noise: The noise parameter
* @param rng: The random-number generator used for generating the position in parameter space
*
* @return double Returns the updated loglikelihood
*/
double hmc_sample(double **parameters, gsl_matrix *x_values, gsl_matrix *data_values, double noise, gsl_rng *rng)
{
    double current_sample[DIMENSION];
    double proposed_sample[DIMENSION];
    double current_momentum[DIMENSION];
    double proposed_momentum[DIMENSION];
    int invertSuccess;
    double loglikelihood;
    
    //Initializing all the matrices, the log-likelihood and the gradient of the log-likelihood.
    double len = data_values->size1;
    gsl_matrix *gradient = gsl_matrix_alloc(2,1);
    gsl_matrix *pureCovMat = gsl_matrix_alloc(len, len);
    gsl_matrix *noiseMat = gsl_matrix_alloc(len, len);
    gsl_matrix *sumMat = gsl_matrix_alloc(len, len);
    gsl_matrix *initInvMat = gsl_matrix_alloc(len, len);
    gsl_matrix *testMatrix = gsl_matrix_alloc(len, len);

    // Initialize current sample
    for (int j = 0; j < RETRY; j++)
    {
        for (int i = 0; i < DIMENSION; i++) 
        {
            current_sample[i] = (*parameters)[i]; // Initialize parameters as last parameters
            current_momentum[i] = gsl_ran_gaussian(rng, 1.0); // Initialize momentum from standard normal distribution
        }

        cov_matrix(pureCovMat, x_values, x_values, *parameters);
        gsl_matrix_memcpy(sumMat, pureCovMat);
        
        gsl_matrix_set_identity(noiseMat);
        gsl_matrix_scale(noiseMat, noise);
        gsl_matrix_add(sumMat, noiseMat);
        invertSuccess = invert_matrix(sumMat, &initInvMat);
        gsl_matrix_set_identity(testMatrix);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, sumMat, initInvMat, -1.0, testMatrix);
        if (gsl_matrix_max(testMatrix) > 1e-7)
        {
            printf("Issues with inverse matrix");
        }
        
        loglikelihood = GPR_logLikelihood(sumMat, initInvMat, data_values);
        
        if ((invertSuccess == 1) && !(isnan(loglikelihood))) {break;}
        if (j == RETRY - 1) {printf("Max tries\n");}
    }

    GPR_loglikelihoodGrad(&gradient, pureCovMat, initInvMat, x_values, data_values, *parameters);
    
    // Save current sample and momentum
    for (int j = 0; j < DIMENSION; j++) {
        proposed_sample[j] = current_sample[j];
        proposed_momentum[j] = current_momentum[j];
    }
    double oldLogLikelihood = loglikelihood;
    
    // Hamiltonian dynamics simulation using leapfrog integration
    
    // Initial half-step for momentum
    for (int i = 0; i < DIMENSION; i++) {
        proposed_momentum[i] -= 0.5 * STEP_SIZE * gsl_matrix_get(gradient, i, 0);
    }
    for (int i = 0; i < NUM_STEPS; i++) {
        
        // Perform leapfrog integration
        leapfrog(&proposed_sample, &proposed_momentum, &loglikelihood, &gradient, &pureCovMat, &initInvMat, noiseMat, x_values, data_values);
    }

    // Final half-step for momentum
    for (int i = 0; i < DIMENSION; i++) {
        proposed_momentum[i] -= 0.5 * STEP_SIZE * gsl_matrix_get(gradient, i, 0);
    
    // Metropolis-Hastings acceptance step
    double current_H = kinetic_energy(current_momentum) + oldLogLikelihood;
    double proposed_H = kinetic_energy(proposed_momentum) + loglikelihood;
    double acceptance_prob = exp(current_H - proposed_H);
    if ((gsl_rng_uniform(rng) < acceptance_prob) && !(isnan(loglikelihood))) {
        // Accept proposed sample
        for (int j = 0; j < DIMENSION; j++) {
            current_sample[j] = proposed_sample[j];
            current_momentum[j] = proposed_momentum[j];
        }
    }
    
    // Save the final sample
    for (int i = 0; i < DIMENSION; i++) {
        (*parameters)[i] = current_sample[i];
    }
    
    gsl_matrix_free(gradient);
    gsl_matrix_free(pureCovMat);
    gsl_matrix_free(noiseMat);
    gsl_matrix_free(sumMat);
    gsl_matrix_free(initInvMat);
    gsl_matrix_free(testMatrix);

    return loglikelihood;
}

/**
* @brief Makes a prediction on the Szz-values for the given coordinates based on a Gaussian process
* 
* @param prediction: A pointer to a pointer to a gsl_matrix containing the prediction values
* @param pred_coords: A pointer to a gsl_matrix containing the preduction coordinate values
* @param data_values: A pointer to a gsl_matrix containing the corresponding Szz values
* @param data_coords: A pointer to a gsl_matrix containing the coordinate values
* @param len: The length of the datalist
* @param predLen: The length of the prediction datalist
* @param parameters: An array containing the parameters
* @param noise: The noise parameter
*
*/
void GPR_predict(gsl_matrix** prediction, gsl_matrix **postCovMat, gsl_matrix* pred_coords, gsl_matrix* data_values, gsl_matrix* data_coords, double* parameters, double noise)
{
    int len = data_coords->size1;
    int predLen = pred_coords->size1;
    int invertSuccess;
    if (len > 0 && predLen > 0)
    {
        gsl_matrix *tempMat = gsl_matrix_alloc(predLen, len);
        gsl_matrix *transMat = gsl_matrix_alloc(len, predLen);
        gsl_matrix *pureCovMat = gsl_matrix_alloc(len, len);
        gsl_matrix *noiseMat = gsl_matrix_alloc(len, len);
        gsl_matrix *sumMat = gsl_matrix_alloc(len, len);
        gsl_matrix *invInitCov = gsl_matrix_alloc(len, len);
        gsl_matrix *mixCovMat = gsl_matrix_alloc(predLen, len); //Note that this is actually the transposted mixed covariance matrix
        gsl_matrix *logValues = gsl_matrix_alloc(len, 1);

        for (int i = 0; i < len; i++)
        {
            gsl_matrix_set(logValues, i, 0, log1p(gsl_matrix_get(data_values, i, 0)));
        }

        gsl_matrix_set_identity(noiseMat); //Noise matrix becomes the identity.
        gsl_matrix_scale(noiseMat, noise); //Noise matrix is multiplied with the noise.

        cov_matrix(pureCovMat, data_coords, data_coords, parameters);
        cov_matrix(mixCovMat, pred_coords, data_coords, parameters);
        cov_matrix(*postCovMat, pred_coords, pred_coords, parameters); //Saves the prediction covariance in the posterior covariance matrix for later use
        
        gsl_matrix_memcpy(sumMat, pureCovMat);
        gsl_matrix_add(sumMat, noiseMat);

        invertSuccess = invert_matrix(sumMat, &invInitCov);
        if (invertSuccess)
        {
            invertSuccess = test_inverse(sumMat, invInitCov, DEFAULT_PRECISION);
            if (!invertSuccess)
            {
                printf("Not inverted correctly\n");
            }
        }
        else
        {
            printf("Inversion failed\n");
        }

        // gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, invInitCov, data_values, 0.0, tempVec); //multiplies the inverted matrix with the Szz-values and saves in tempVec
        // gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, mixCovMat, tempVec, 0.0, *prediction); //multiplies the mixed covariance matrix with tempVec from before and saves as prediction

        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, mixCovMat, invInitCov, 0.0, tempMat); //multiplies the mixed covariance matrix with the inverted matrixx and saves in tempMat
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tempMat, logValues, 0.0, *prediction); //multiplies tempMat with the data values and saves as prediction
        
        for (int i = 0; i < predLen; i++)
        {
            gsl_matrix_set(*prediction, i, 0, exp(gsl_matrix_get(*prediction, i, 0) - 1));
        }

        gsl_matrix_transpose_memcpy(transMat, mixCovMat); //Transposes the already transposed mixed matrix to get the covariance
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, tempMat, transMat, 1.0, *postCovMat); //subtracts the tempMat multiplied on the mixed covariance matrix from the prediction covariance and saves in the posterior matrix



        gsl_matrix_free(pureCovMat);
        gsl_matrix_free(noiseMat);
        gsl_matrix_free(sumMat);
        gsl_matrix_free(invInitCov);
        gsl_matrix_free(mixCovMat);
        gsl_matrix_free(tempMat);
        gsl_matrix_free(logValues);
    }
    else
    {
        printf("Issue with lengths\n");
    }
}

void GPR_single_initialize(gsl_matrix** inv_init_cov, gsl_matrix* noiseMat, gsl_matrix* data_coords, double* parameters)
{
    int len = data_coords->size1;
    int invertSuccess;

    gsl_matrix *pureCovMat = gsl_matrix_alloc(len, len);
    gsl_matrix *sumMat = gsl_matrix_alloc(len, len);
    
    cov_matrix(pureCovMat, data_coords, data_coords, parameters);        
    gsl_matrix_memcpy(sumMat, pureCovMat);
    gsl_matrix_add(sumMat, noiseMat);

    invertSuccess = invert_matrix(sumMat, inv_init_cov);
    if (invertSuccess)
    {
        invertSuccess = test_inverse(sumMat, *inv_init_cov, DEFAULT_PRECISION);
        if (!invertSuccess)
        {
            printf("Not inverted correctly\n");
        }
    }
    else
    {
        printf("Inversion failed\n");
    }
}

void GPR_single_predict(gsl_matrix** prediction, gsl_matrix **post_cov_mat, gsl_matrix **inv_init_cov, gsl_matrix* pred_coords, gsl_matrix* data_values, gsl_matrix* data_coords, double* parameters, double noise, bool *first_time)
{
    int len = data_coords->size1;
    int predLen = pred_coords->size1;
    int invertSuccess;
    FILE *filePointer;
    if (len > 0 && predLen > 0)
    {
        // gsl_matrix *tempVec = gsl_matrix_alloc(len, 1);
        gsl_matrix *tempMat = gsl_matrix_alloc(predLen, len);
        gsl_matrix *regressionValues = gsl_matrix_alloc(len, 1);
        gsl_matrix *transMat = gsl_matrix_alloc(len, predLen);
        gsl_matrix *pureCovMat = gsl_matrix_alloc(len, len);
        gsl_matrix *noiseMat = gsl_matrix_alloc(len, len);
        gsl_matrix *sumMat = gsl_matrix_alloc(len, len);
        gsl_matrix *mixCovMat = gsl_matrix_alloc(predLen, len); //Note that this is actually the transposted mixed covariance matrix

        gsl_matrix_set_identity(noiseMat); //Noise matrix becomes the identity.
        gsl_matrix_scale(noiseMat, noise); //Noise matrix is multiplied with the noise.
        
        cov_matrix(mixCovMat, pred_coords, data_coords, parameters);
        cov_matrix(*post_cov_mat, pred_coords, pred_coords, parameters);
    
        if (*first_time)
        {
            GPR_single_initialize(inv_init_cov, noiseMat, data_coords, parameters);
            *first_time = false;
        }
        
        for (int i = 0; i < len; i++)
        {
            gsl_matrix_set(regressionValues, i, 0, log1p(gsl_matrix_get(data_values, i, 0)));
        }

        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, mixCovMat, *inv_init_cov, 0.0, tempMat); //multiplies the mixed covariance matrix with the inverted matrix and saves in tempMat
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tempMat, regressionValues, 0.0, *prediction); //multiplies tempMat with the data values and saves as prediction

        gsl_matrix_transpose_memcpy(transMat, mixCovMat); //Transposes the already transposed mixed matrix to get the covariance
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, tempMat, transMat, 1.0, *post_cov_mat); //multiplies the mixed covariance matrix with tempVec from before and saves as prediction

        for (int i = 0; i < predLen; i++)
        {
            gsl_matrix_set(*prediction, i, 0, exp(gsl_matrix_get(*prediction, i, 0) - 1));
        }



        // gsl_matrix_free(tempVec);
        gsl_matrix_free(pureCovMat);
        gsl_matrix_free(noiseMat);
        gsl_matrix_free(regressionValues)
        gsl_matrix_free(sumMat);
        gsl_matrix_free(mixCovMat);
        gsl_matrix_free(tempMat);
        gsl_matrix_free(transMat);
    }
    else
    {
        printf("Issue with lengths\n");
    }
}


/**
* @brief Calculates the value of the normalized gaussian function at a given distance, for a given σ
* 
* @param distance_squared: The squared distance
* @param sigma: The spread of the gaussian function
*
* @return: double Returns the value of the gaussian
*/
double gaussianFunc(double distance_squared, double sigma)
{
    return exp(-distance_squared/(2 * pow(sigma, 2)))/(sqrt(2 * M_PI) * sigma);
}

/**
* @brief Uses a set of gaussian distributions centered at the known data values to make a prediction on the Szz-values. 
* 
* @param prediction: A pointer to a pointer to a gsl_matrix containing the prediction values
* @param pred_coords: A pointer to a gsl_matrix containing the preduction coordinate values
* @param data_values: A pointer to a gsl_matrix containing the corresponding Szz values
* @param data_coords: A pointer to a gsl_matrix containing the coordinate values
* @param len: The length of the datalist
* @param predLen: The length of the prediction datalist
* @param parameters: An array containing the parameters
*
*/
void Gauss_predict(gsl_matrix** prediction, gsl_matrix* pred_coords, gsl_matrix* data_values, gsl_matrix* data_coords, double* parameters)
{
    int len = data_coords->size1;
    int predLen = pred_coords->size1;
    double distance_squared;
    double oldPred;
    printf("Length and prediction length: %d %d\n", len, predLen);
    for (size_t k = 0; k < pred_coords->size1; k++)
    {
        gsl_matrix_set(*prediction, k, 0, 0.0);
        for (size_t i = 0; i < data_values->size1; i++)
        {
            //Calculating the squared distance between point k and i
            distance_squared = 0.0;
            for (size_t j = 0; j < DIMENSION; j++)
            {
                distance_squared += pow(gsl_matrix_get(pred_coords, k, j) - gsl_matrix_get(data_coords, i, j), 2);
            }
            oldPred = gsl_matrix_get(*prediction, k, 0);
            gsl_matrix_set(*prediction, k, 0, oldPred + gsl_matrix_get(data_values, i, 0) * gaussianFunc(distance_squared, parameters[0]));
        }
    }
}

/**
* @brief Uses a set of gaussian distributions centered at the known data values to make a prediction on the Szz-values. 
    This function varies the spread of the gaussians based on the shortest distance to surrounding datapoints.
* 
* @param prediction: A pointer to a pointer to a gsl_matrix containing the prediction values
* @param pred_coords: A pointer to a gsl_matrix containing the preduction coordinate values
* @param data_values: A pointer to a gsl_matrix containing the corresponding Szz values
* @param data_coords: A pointer to a gsl_matrix containing the coordinate values
* @param len: The length of the datalist
* @param predLen: The length of the prediction datalist
* @param parameters: An array containing the parameters
*
*/
void advanced_Gauss_predict(gsl_matrix** prediction, gsl_matrix* pred_coords, gsl_matrix* data_values, gsl_matrix* data_coords, double* parameters)
{
    int index;
    int len = data_coords->size1;
    int predLen = pred_coords->size1;
    double *data_distance_squared = (double*)malloc(data_coords->size1*sizeof(double));
    double sigmaVal;
    double pred_distance_squared;
    double minDist;
    double maxDist;
    double predTerm;
    double oldPred;
    for (size_t k = 0; k < data_coords->size1; k++)
    {
        minDist  = 0.0;
        maxDist  = 0.0;
        for (size_t i = 0; i < data_coords->size1; i++)
        {
            //Calculating the squared distance between point k and i for non-diagonal elements
            data_distance_squared[i] = 0.0;
            for (size_t j = 0; j < DIMENSION; j++)
            {
                data_distance_squared[i] += pow(gsl_matrix_get(data_coords, k, j) - gsl_matrix_get(data_coords, i, j), 2);
            }
        }
        //We dont want the sigma to appear when points are essentially atop of each other
        find_min_max(data_distance_squared, data_coords->size1, &minDist, &maxDist);
        while (minDist < DEFAULT_PRECISION)
        {
            index = find_index(data_distance_squared, data_coords->size1, minDist);
            data_distance_squared[index] = maxDist;
            find_min_max(data_distance_squared, data_coords->size1, &minDist, &maxDist);
        }
        printf("Minimal distance %f\n", minDist);
        sigmaVal = sqrt(minDist) * parameters[0]; //Taking the square root of the minimal distance and making sigma proportional to this
        for (size_t l = 0; l < pred_coords->size1; l++)
        {
            pred_distance_squared = 0.0;
            for (size_t j = 0; j < DIMENSION; j++)
            {
                pred_distance_squared += pow(gsl_matrix_get(data_coords, k, j) - gsl_matrix_get(pred_coords, l, j), 2);
            }

            predTerm = gsl_matrix_get(data_values, k, 0) * gaussianFunc(pred_distance_squared, sigmaVal);
            oldPred = gsl_matrix_get(*prediction, l, 0);
            gsl_matrix_set(*prediction, l, 0, oldPred + predTerm);
        }
    }
    free(data_distance_squared);
}

double normalizationCoef(gsl_matrix *prediction, int N, double dk, double dE)
{
    double norm = 0.0;
    double idealNorm = 3/2 * M_PI * N;
    for (int i = 0; i<prediction->size1; i++)
    {
        norm += gsl_matrix_get(prediction, i, 0)*dk*dE;
    }
    // norm /= prediction->size1;

    return idealNorm/norm;
}

/**
 * @brief Reads data from a szz-file into a q-array, an E-array and a Szz-array, all having the same length and renormalizes the q- and szz-values.
 * 
 * @param filename The name of the file containing the energies and Szz-values.
 * @param qArray A dynamically allocated array, that will contain the q-values, corresponding to each (hw,Szz)-set.
 * @param eArray A dynamically allocated array, that will contain the energies.s
 * @param szzArray A dynamically allocated array, that will contain the Szz-values.
 * @return int Returns 0 if the file could not be opened, the length of the arrays otherwise.
 */
int readFile(const char *filename, double **qArray, double **eArray, double **szzArray)
{
    FILE *file_pointer;
    char line[200]; // Assuming each line in the file is no longer than 200 characters
    char *char_ptr;
    char *char_ptr_end;

    int appendCount = 0;
    int qCount = 0;
    double totSzz = 0;

    double *qList;
    double *eList;
    double *szzList;

    // Open the file in read mode
    file_pointer = fopen(filename, "r");

    // Check if the file opened successfully
    if (file_pointer == NULL)
    {
        printf("Error opening file.\n");
        return 0;
    }

    int lineCount = 0;
    char ch;

    while ((ch = fgetc(file_pointer)) != EOF)
    {
        if (ch == '\n')
        {
            lineCount++;
        }
    }
    // Rewinding the file to read from it again.
    rewind(file_pointer);

    // Dynamical memory allocation to reduce it later
    qList = (double*)malloc(lineCount * sizeof(double));
    eList = (double*)malloc(lineCount * sizeof(double));
    szzList = (double*)malloc(lineCount * sizeof(double));

    // Read each line of the file
    while (fgets(line, sizeof(line), file_pointer) != NULL)
    {

        // reading of q-elements
        char_ptr = strchr(line, 'q');
        double qVal;
        char qText[200] = "";
        // If the string contains "q=", it will read off the q-value
        if ((char_ptr != NULL) && (*(char_ptr + 1) == '='))
        {
            char_ptr_end = strchr(char_ptr + 4, ' ');
            for (char *ptr = (char_ptr + 4); ptr < char_ptr_end; ptr++)
            {
                char temp[2] = {*ptr, '\0'}; // Temporary string for the current character
                strncat(qText, temp, 1);
            }
            char *endptr = NULL;
            long int qAttempt = strtol(qText, &endptr, 10);

            if (*endptr != '\0' || endptr == qText)
            {
                printf("Error: qText is not a valid integer\n");
            }
            else
            {
                qVal = (double)qAttempt;
                qCount++;
            }
        }
        else // Now we look for the lines only containing E- and Szz-values
        {
            double eVal;
            char eText[200] = "";
            char *E_ptr = line;

            double szzVal;
            char szzText[200] = "";
            while (*E_ptr == ' ') // Skip leading spaces
            {
                E_ptr++;
            }
            char *Szz_ptr = strchr(E_ptr, ' '); // The next space is just before the Szz-values

            // Reading the energies
            for (char *ptr = E_ptr; ptr < Szz_ptr; ptr++)
            {
                char temp[2] = {*ptr, '\0'}; // Temporary string for the current character
                strncat(eText, temp, 1);
            }
            char *endptr = NULL;
            double eAttempt = strtod(eText, &endptr);

            if (*endptr != '\0' || endptr == eText)
            {
                printf(""); //This is a bit dumb, but not a priority
            }
            else
            {

                // Reading the Szz-values
                char *lineEnd_ptr = strchr(Szz_ptr, '\0');
                if (lineEnd_ptr == NULL)
                {
                    lineEnd_ptr = strchr(Szz_ptr, '\n');
                }
                for (char *ptr = Szz_ptr + 1; ptr < lineEnd_ptr; ptr++)
                {
                    char temp[2] = {*ptr, '\0'}; // Temporary string for the current character
                    strncat(szzText, temp, 1);
                }
                endptr = NULL;

                eVal = eAttempt;
                szzVal = strtod(szzText, &endptr);
                qList[appendCount] = qVal;
                eList[appendCount] = eVal;
                szzList[appendCount] = szzVal;
                appendCount++;
            }
        }
    }

    // Close the file
    fclose(file_pointer);

    //Renormalizing q-space:
    for (int i = 0; i < appendCount; i++)
    {
        qList[i] /= qCount;
    }

    //Renormalizing Szz-space:
    for (int i = 0; i < appendCount; i++)
    {
        totSzz += szzList[i];
    }

    for (int i = 0; i < appendCount; i++)
    {
        szzList[i] *= qCount/totSzz; //qCount is equivalent to the number of atoms in the simulation, so now it sums up to the number of atoms.
    }

    *qArray = (double*)realloc(qList, appendCount * sizeof(*qList));
    *eArray = (double*)realloc(eList, appendCount * sizeof(*eList));
    *szzArray = (double*)realloc(szzList, appendCount * sizeof(*szzList));

    if (*qArray == NULL || *eArray == NULL || *szzArray == NULL)
    {
        printf("Reallocation failed, double check arrays\n");
        *qArray = qList;
        *eArray = eList;
        *szzArray = szzList;
        free(qList);
        free(eList);
        free(szzList);
    }

    return appendCount;
}

/**
 * @brief Reads data from multiple szz-files into a q-array, an E-array and a Szz-array, all having the same length.
 * 
 * @param filenames An array containing the names of the files with the q-values, energies and Szz-values.
 * @param qArray A dynamically allocated array, that will contain the q-values, corresponding to each (hw,Szz)-set.
 * @param eArray A dynamically allocated array, that will contain the energies.
 * @param szzArray A dynamically allocated array, that will contain the Szz-values.
 * @return int Returns 0 if the file could not be opened, the length of the arrays otherwise.
 */
int multiReadFile(const char **filenames, size_t nFiles, double **qArray, double **eArray, double **szzArray)
{
    int totLen = 0;
    double *qTemp = NULL;
    double *eTemp = NULL;
    double *szzTemp = NULL;
    *qArray = NULL;
    *eArray = NULL;
    *szzArray = NULL;

    int addLen;
    double minQ;
    double maxQ;
    for (size_t i = 0; i < nFiles; i++)
    {
        addLen = readFile(filenames[i], &qTemp, &eTemp, &szzTemp);
        *qArray = (double*)realloc(*qArray, (totLen + addLen) * sizeof(double));
        *eArray = (double*)realloc(*eArray, (totLen + addLen) * sizeof(double));
        *szzArray = (double*)realloc(*szzArray, (totLen + addLen) * sizeof(double));

        if (*qArray == NULL || *eArray == NULL || *szzArray == NULL)
        {
            printf("Reallocation failed, double check arrays\n");
            free(qTemp);
            free(eTemp);
            free(szzTemp);
            return -1;
        }
        
        find_min_max(qTemp, addLen, &minQ, &maxQ); //As the files apparently vary in whether it starts with zero or -maxQ, these are found
        for (int j = 0; j < addLen; j++)
        {
            (*qArray)[totLen + j] = qTemp[j] - minQ; //minQ is subtracted - thus all q-arrays start at zero
            (*eArray)[totLen + j] = eTemp[j];
            (*szzArray)[totLen + j] = szzTemp[j];
        }
        free(qTemp);
        free(eTemp);
        free(szzTemp);
        totLen += addLen;
    }
    return totLen;
}

/**
 * @brief Writes the prediction values and coordinates to a file
 * 
 * @param predFilename The string name of the file where the prediction will be saved. Needs not to exist and will be overwritten if it does
 * @param coords The q and E values corresponding to the Szz-values
 * @param values The Szz values
 */
void writePredFile(const char *predFilename, gsl_matrix *coords, gsl_matrix *values, gsl_matrix *covariance)
{
    FILE *file_pointer;
    file_pointer = fopen(predFilename, "w");

    if (file_pointer == NULL)
    {
        printf("Error opening file\n");
        return;
    }

    for (size_t i = 0; i < coords->size1; i++)
    {
        if (i == 0)
        {
            fprintf(file_pointer, "q hw Szz uncer\n");    
        }
        
        if (gsl_matrix_get(values, i, 0) < 100) //Hard removal of infinite points
        {
            fprintf(file_pointer, "%f %f %f %f\n", gsl_matrix_get(coords, i, 0), gsl_matrix_get(coords, i, 1), gsl_matrix_get(values, i, 0), sqrt(gsl_matrix_get(covariance, i, i)));
        }
    }

    fclose(file_pointer);
}

int main()
{
    char *testPtr = NULL;

    double *Q;
    double *E;
    double *Szz;
    double logLikelihood[NUM_SAMPLES];
    double dk;
    double dE;

    //Creating the constant array of strings, containing all the filenames.
    glob_t glob_result;
    glob(pattern, GLOB_TILDE, NULL, &glob_result);
    size_t namesLen = glob_result.gl_pathc;
    const char *filenames[namesLen];
    for (size_t i = 0; i < namesLen; i++)
    {
        filenames[i] = glob_result.gl_pathv[i];
    }
    
    int len = multiReadFile(filenames, namesLen, &Q, &E, &Szz);
    printf("lenght is %d\n",len);
    predLen = len;
    if (predInt == 0)
    {
        predLen = 1;
    }
    
    if (predInt == 2)
    {
        predLen = nk*nE;
    }

    gsl_matrix *prediction = gsl_matrix_alloc(predLen, 1);
    gsl_matrix *postCovariance = gsl_matrix_alloc(predLen, predLen);
    gsl_matrix *predCoords = gsl_matrix_alloc(predLen, DIMENSION);
    gsl_matrix *values = gsl_matrix_alloc(len, 1);
    gsl_matrix *coords = gsl_matrix_alloc(len, DIMENSION);
    gsl_matrix *covMat = gsl_matrix_alloc(len, len);
    gsl_matrix *initInvMat = gsl_matrix_alloc(len, len);

    for (int i = 0; i < len; i++)
    {
        gsl_matrix_set(coords, i, 0, Q[i]);
        gsl_matrix_set(coords, i, 1, E[i]);
        gsl_matrix_set(values, i, 0, Szz[i]);
    }

    if (predInt == 0)
    {
        for (size_t d = 0; d < DIMENSION; d++)
        {
            gsl_matrix_set(predCoords, 0, d, singleCoords[d]);
        }
    }

    if (predInt == 1)
    {
        gsl_matrix_memcpy(predCoords, coords);
    }

    if (predInt == 2)
    {
        dk = (predMax_k - predMin_k)/nk;
        dE = (predMax_E - predMin_E)/nE;
        for (int i = 0; i < nE*nk; i++)
        {
            gsl_matrix_set(predCoords, i, 0, predMin_k + (i%nk) * dk); //Running through the k-values such that every nk index returns to the first k-value
            gsl_matrix_set(predCoords, i, 1, predMin_E + (i/nk) * dE); //Running through the E-values such that every time k resets, it uses a new E-value
        }
    }

    //TRAINING GROUNDS!!!!!
    if (train)
    {
        FILE *paramFile;
        paramFile = fopen("HMCparam.txt", "w");
        if (paramFile == NULL)
        {
            printf("Error opening parameter file\n");
            return -1;
        }
        
        gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
        gsl_rng_set(rng, time(NULL)); // Seed the random number generator with the current time

        double param1;
        double param2;
        clock_t start;
        clock_t end;
        double time_spent;
        double total_time;
        double sumLikelihood;
        double *sample = (double*)malloc(DIMENSION*sizeof(double));
        gsl_matrix *logData = gsl_matrix_alloc(len, 1);
        int frac = NUM_SAMPLES/100;

        for (int i = 0; i < len; i++)
        {
            gsl_matrix_set(logData, i, 0, log1p(gsl_matrix_get(values, i, 0)));
        }
        

        if (sample == NULL) 
            {
                printf("Allocation of samples failed\n");
                return -1;
            }

        // Generate samples for HMC
        int nSpaces = 150;
        char clean[nSpaces + 1];
        memset(clean, ' ', nSpaces); // Fill the first 200 characters of clean with spaces
        clean[nSpaces] = '\0';       // Null-terminate the string
        printf("Initial parameters : %f %f\n", param[0], param[1]);
        for (int i = 0; i < NUM_SAMPLES; i++)
        {
            printf("%s\r", clean); //cleans the line and returns to the beginning
            
            printf("Progress: [");
            for (int j = 0; j <= i/frac; j++)
            {
                printf("=");
            }
            for (int j = i/frac + 1; j < 100; j++) {
                printf(" ");
            }
            if (i/frac - 1 < 100)
            {
                if (i == 0)
                {
                    printf("] %d/%d \r", i/frac + 1, 100);
                }
                else
                {
                    printf("] %d/%d, %.1f minutes left\r", i/frac + 1, 100, total_time - time_spent * i);
                }
                fflush(stdout);
            }

            for (int j = 0; j < DIMENSION; j++)
            {
                sample[j] = param[j];
            }
            
            if (i==0)
            {
                start = clock();
            }
            
            logLikelihood[i] = hmc_sample(&sample, coords, logData, noise, rng);
            
            if (i==0)
            {
                end = clock();
                time_spent = (double)(end - start) / CLOCKS_PER_SEC / 60;
                total_time = time_spent * NUM_SAMPLES;
            }
            
            if (isinf(logLikelihood[i]) || isnan(logLikelihood[i])) {logLikelihood[i] = minLogVal;}
            
            if (i == 0) {fprintf(paramFile, "logP s l\n");}
            
            param1 += pow(sample[0],2)*exp(logLikelihood[i]); //Takes the squared value to make it a single-peak distribution
            param2 += pow(sample[1],2)*exp(logLikelihood[i]);
            sumLikelihood += exp(logLikelihood[i]);
            fprintf(paramFile, "%f %f %f\n",logLikelihood[i], sample[0], sample[1]); //Generalize to DIMENSION in stead of two I guess?
        }
        printf("Progress:\n");
        fclose(paramFile);


        // Output the samples
        double minLikelihood;
        double maxLikelihood;
        double relLikelihood = mean(logLikelihood, NUM_SAMPLES);
        find_min_max(logLikelihood, NUM_SAMPLES, &minLikelihood, &maxLikelihood);
        param[0] = sqrt(param1)/sqrt(sumLikelihood);
        param[1] = sqrt(param2)/sqrt(sumLikelihood);
        printf("Expectation values: %f, %f\n", param[0], param[1]);
        printf("Mean, max and min log-likelihood: %f %f %f\n", relLikelihood, maxLikelihood, minLikelihood);

        free(sample);
        gsl_rng_free(rng);
        gsl_matrix_free(logData);
    }

    if (trainedBefore)
    {
        FILE *paramFile;
        char *char_ptr;
        char *testPtr;
        char *char_ptr_end;
        char line[200];
        double likelihood;
        double sVal;
        double lVal;
        char *copy;
        double sumLikelihood = 0.0;
        double sumS = 0.0;
        double sumL = 0.0;
        
        paramFile = fopen("HMCparam.txt", "r");
        if (paramFile == NULL)
        {
            printf("Error opening parameter file\n");
            return -1;
        }
        
        if (fgets(line, sizeof(line), paramFile) == NULL) //Checks if file is empty and skips first line!
        {
            // Handle the error if the file is empty
            fprintf(stderr, "File is empty\n");
            return -1;
        }

        while (fgets(line, sizeof(line), paramFile) != NULL)
        {
            char_ptr = line;
            char likeliText[200] = "";
            char sText[200] = "";
            char lText[200] = "";
            char_ptr_end = strchr(char_ptr, ' ');

            for (char *ptr = char_ptr; ptr < char_ptr_end; ptr++)
            {
                char temp[2] = {*ptr, '\0'}; // Temporary string for the current character
                strncat(likeliText, temp, 1);
            }

            testPtr = NULL;
            likelihood = strtod(likeliText, &testPtr);

            char_ptr = char_ptr_end + 1;
            char_ptr_end = strchr(char_ptr, ' ');

            for (char *ptr = char_ptr; ptr < char_ptr_end; ptr++)
            {
                char temp[2] = {*ptr, '\0'}; // Temporary string for the current character
                strncat(sText, temp, 1);
            }
            
            testPtr = NULL;
            sVal = strtod(sText, &testPtr);

            char_ptr = char_ptr_end + 1;
            char_ptr_end = strchr(char_ptr, '\n');
            if (char_ptr_end == NULL)
            {
                char_ptr_end = strchr(char_ptr, '\0');
            }

            for (char *ptr = char_ptr; ptr < char_ptr_end; ptr++)
            {
                char temp[2] = {*ptr, '\0'}; // Temporary string for the current character
                strncat(lText, temp, 1);
            }
            
            testPtr = NULL;
            lVal = strtod(lText, &testPtr);

            sumS += pow(sVal,2) * exp(likelihood);
            sumL += pow(lVal,2) * exp(likelihood);
            sumLikelihood += exp(likelihood);
        }
        param[0] = sqrt(sumS)/sqrt(sumLikelihood);
        param[1] = sqrt(sumL)/sqrt(sumLikelihood);
        printf("Expectation values: %f, %f\n", param[0], param[1]);
        fclose(paramFile);
    }
    if (predict)
    {
        if (predInt == 0)
        {
            GPR_single_predict(&prediction, &postCovariance, &initInvMat, predCoords, values, coords, param, noise, &firstTime);
            print_matrix(prediction);
        }
        else
        {
            
            GPR_predict(&prediction, &postCovariance, predCoords, values, coords, param, noise);
            double normCoef = normalizationCoef(prediction, 18, dk, dE);
            printf("Norm: %f\n", normCoef);
            writePredFile(predFile, predCoords, prediction, postCovariance);
        }
    // advanced_Gauss_predict(&prediction, predCoords, values, coords, param);
    // Gauss_predict(&prediction, predCoords, values, coords, param);
    }

    free(Q);
    free(E);
    free(Szz);
    gsl_matrix_free(prediction);
    gsl_matrix_free(values);
    gsl_matrix_free(coords);
    gsl_matrix_free(predCoords);
    gsl_matrix_free(covMat);
    gsl_matrix_free(postCovariance);
    gsl_matrix_free(initInvMat);
    globfree(&glob_result);
    return 0;
}
