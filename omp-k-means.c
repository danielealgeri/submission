#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

int n_dims;             /* number of dimensions.                */

int n_points;           /* number of data points.               */

int n_clusters;         /* number of clusters.                  */

float *data;            /* [array of length (n_points * n_dims)]
                           `&data[i*n_dims]` points to the beginning
                           of the i-th data items, which is an array
                           of `n_dims` floating-point numbers.  */

float *centroids;       /* [array of length (n_clusters * n_dims)]
                           `&centroids[j*n_dims]` points to the
                           beginning of the j-th centroid, which is an
                           array of `n_dims` floating point
                           numbers.                             */

float *new_centroids;   /* [array of length (n_clusters * n_dims)] */

int *counts;            /* [array of length n_clusters] `counts[j]`
                           is the number of points that belong to
                           cluster j.                           */

int *cluster_of;        /* [array of length n_points] `clusters_of[i]`
                           is the ID of the cluster assigned to the
                           i-th data point; cluster IDs are integer in
                           0..(n_clusters-1).                   */


/* A safe version of `malloc()` that aborts if memory allocation
   fails. */
void *safe_malloc(size_t size)
{
    void *result = malloc(size);
    assert(result != NULL);
    return result;
}

/******************************************************************************
 **
 ** Utility functions that operate on arrays of `n_dims` elements.
 **
 ******************************************************************************/

/* Set all components of vector `p` of size `n_dims` equal to zero. */
void vzero( float *p )
{
    for (int d=0; d<n_dims; d++)
        p[d] = 0.0f;
}

/* Add vector `p1` to vector `p2`; store result in `p1`. Both vectors
   have size `n_dims`. */
void vadd( float *p1, const float *p2 )
{
    for (int d=0; d<n_dims; d++)
        p1[d] += p2[d];
}

/* Multiply each element of vector `p` of size `n_dims` by `v`. */
void vmul( float *p, float v )
{
    for (int d=0; d<n_dims; d++)
        p[d] *= v;
}

/* Copy `p2` into `p1`. */
void vcopy( float *p1, const float *p2 )
{
    for (int d=0; d<n_dims; d++)
        p1[d] = p2[d];
}

/* Compute the Euclidean squared distance of `p1` and `p2`. */
float sqdist( float *p1, float *p2 )
{
    float result = 0.0;
    for (int d=0; d<n_dims; d++) {
        result += (p1[d] - p2[d])*(p1[d] - p2[d]);
    }
    return result;
}

/******************************************************************************
 **
 ** K-Means algorithm begins here.
 **
 ******************************************************************************/

/* This function can be used to access the arrays (actually, matrices)
   `data`, `centroids` and `new_centroids`. These are all matrices
   with `n_dims` columns. The function returns the linear index of row
   `i` and column `d`. Example: `data[IDX(i, d)]` is equivalent to
   `data[i*n_dims + d]`. */
int IDX(int i, int d)
{
    return i*n_dims + d;
}

/* Return a random integer in a..b. This function must not be
   parallelized, since `rand()` is not thread-safe. */
int randab(int a, int b)
{
    return a + rand() % (b-a+1);
}

/* Centroids are initialized by randomly selecting `n_clusters` data
   points. To select `n_clusters` out of `n_data` elements, we use
   Knuths' algorithm as reported in J. Bentley, "Programming Pearls",
   2nd ed., Addison-Wesley, 2000, p. 126.

   DO NOT PARALLELIZE THIS FUNCTION: `rand()` is not thread-safe. */
void init_centroids( void )
{
    int select = n_clusters;
    int remaining = n_points;
    for (int i=0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            /* Select point `i` as one of the centroids. */
            vcopy( &centroids[IDX(select,0)], &data[IDX(i,0)] );
        }
        remaining--;
    }
}

/* Assign each data point to the nearest centroid. Updates
   the `counts` array. */
void classify( void )
{
    /*Sequential Initialization*/
    for (int j=0; j<n_clusters; j++) {
        counts[j] = 0;
    }

    /*OpenMP directive to implement the parallelization of the for loop.
    Every iteration of the loop takes care of one single datapoint 'i' in the input dataset.
    Since the operations performed in the boduy of this loop for one datapoint 'i' are completely independent
    from every other datapoint, this for loop can be parallelised. To do this, we use the 'parallel for' directive of OpenMP which
    distributes the iterations across the available threads.
    One of the two writes we perform in the body of the loop does not cause any race condition: any given index of the 'cluster_of' array is associated
    to a specific datapoint, and thus is accessed and written by one iteration only.
    On the other hand, many concurrent threads may possibly try to access the 'counts' array in the same position at the same time, causing a race condition.
    To avoid the race condition, we apply the reduction clause to the array, so that each thread works on its private copy of the array.
    At the end of the loop, such copies are merged together to obtain the final counts[] array.*/
    #pragma omp parallel for default(none) \
            shared(n_points, n_clusters, n_dims, data, centroids, cluster_of) \
            reduction(+: counts[0:n_clusters])
    for (int i=0; i<n_points; i++) {            
        int nearest = 0;
        float mindist = sqdist( &data[IDX(i,0)], &centroids[IDX(nearest, 0)] );
        for (int j=1; j<n_clusters; j++) {
            const float dist = sqdist( &data[IDX(i, 0)], &centroids[IDX(j, 0)] );
            if ( dist < mindist ) {
                mindist = dist;
                nearest = j;
            }
        }
        cluster_of[i] = nearest;
        counts[nearest]++;
    }
}

/* Update the centroids. Set the centroid of each cluster to the
   barycenter of the points. Returns the maximum shift, i.e., the
   maximum difference between the (squared) old and new position of
   all centroids. */
float update_centroids( void )
{
    // Initialize the new_centroids vectror to 0
    for (int j=0; j<n_clusters; j++) {
        vzero( &new_centroids[IDX(j, 0)] );
    }

    /*In a similar fashion to what was done for the classify() method, we parallelize the loop as each iteration is associated to one single, independent datapoint 'i'.
    The reduction clause on the 'new_centroids' array ensures that each threas writes on its own private copy, avoiding race conditions.*/
    #pragma omp parallel for default(none) \
            shared(n_points, n_clusters, n_dims, data, cluster_of) \
            reduction(+:new_centroids[0:n_clusters*n_dims])
    for (int i=0; i<n_points; i++) {
        vadd( &new_centroids[IDX(cluster_of[i], 0)], &data[IDX(i, 0)] );
    }

    float maxshift = 0.0f;
    for (int j=0; j<n_clusters; j++) {
        /* If a cluster is empty, we simply copy the old centroid to
           the new one. */
        if (counts[j] == 0) {
            vcopy( &new_centroids[IDX(j,0)], &centroids[IDX(j,0)] );
        } else {
            // We divide the accumulated sum in new_centroids by the number of points assigned to current cluster
            vmul( &new_centroids[IDX(j, 0)], 1.0f/counts[j] );
        }
        // The max_shift refers to how much the CENTROIDS have overall moved between one iteration and the other
        const float shift = sqdist( &centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)] );
        if (shift > maxshift)
            maxshift = shift;
        vcopy( &centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)] );
    }

    return maxshift;
}

/******************************************************************************
 **
 ** Input/output functions. DO NOT parallelize them.
 **
 ******************************************************************************/

/* Read the input data from `f`. Each row must contain `n_dims`
   numbers. This function figures out how many numbers are in a row,
   and how many rows there are. Then, it initializes the variables
   `n_dims` and `n_points` accordingly. */
void read_input( FILE *f )
{
    const size_t BUFLEN = 1024;
    char buffer[BUFLEN];

    /* Get the first line of the input file, and count how many
       numbers are there. This function is not very robust: if the
       first line is empty, the number of dimensions will be zero; if
       the first line has more than `BUFLEN` characters, the number of
       fields will be computed incorrectly. */
    char *i_dont_care = fgets(buffer, BUFLEN, f);
    (void)i_dont_care; /* Avoid a compiler warning. */
    n_dims = -1;
    char *start, *end = buffer;
    do {
        start = end;
        strtof(start, &end);
        n_dims++;
    } while (end != start);

    assert(n_dims > 0); /* If this assertion fails, then the first
                           line of the input is empty. */

    /* Rewind the file and count how many data items are there. */
    rewind(f);
    int n_items = 0;
    float dummy;
    while (1 == fscanf(f, "%f", &dummy))
        n_items++;

    assert(n_points % n_dims == 0); /* If this assertion fails, then
                                       there is some line of the input
                                       file that has != n_dims
                                       items. */

    n_points = n_items / n_dims;

    data = (float*)safe_malloc(n_points * n_dims * sizeof(*data));

    /* Rewind and read the actual data. */
    rewind(f);
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            const int nread = fscanf(f, "%f", &data[IDX(i, d)]);
            assert(nread == 1);
        }
    }
}

#ifdef MAKE_MOVIE

/* Save the intermediate coordinates of the centroids into a
   file.

   This function is useful for generating a movie showing how the
   centroids get updated, otherwise it can be omitted.

   This function can be enable by defining the MAKE_MOVIE symbol at
   compilation time. */
void save_centroids( int iter )
{
    char buf[1024];

    snprintf(buf, sizeof(buf), "centroids_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w"); assert(f != NULL);
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int j=0; j<n_clusters; j++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", centroids[IDX(j, d)]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

/* Save the intermediate coordinates of the points and their clusters
   into a file.

   This function is useful for generating a movie showing how the
   centroids get updated, otherwise it can be omitted.

   This function can be enable by defining the MAKE_MOVIE symbol at
   compilation time.
*/
void save_clusters( int iter )
{
    char buf[1024];

    snprintf(buf, sizeof(buf), "out_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w");
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
    fclose(f);
}

#endif

/* Print the final result of the computation, i.e, the coordinates of
   the centroids and the list of data points with the cluster id. */
void save_results( FILE *f )
{
    fprintf(f, "# Centroids:\n#\n");
    for (int j=0; j<n_clusters; j++) {
        fprintf(f, "# %3d :", j);
        for (int d=0; d<n_dims; d++) {
            fprintf(f, " %f", centroids[IDX(j, d)]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "#\n");
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
}

/******************************************************************************
 **
 ** Main program.
 **
 ******************************************************************************/
int main( int argc, char *argv[] )
{
    FILE *inputf, *outputf;
    const int MAXITER = 100;
    const float TOL = 1e-5;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s K input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(123); /* Deterministic initialization of the PRNG. */

    n_clusters = atoi(argv[1]);

    if ((inputf = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "FATAL: can not open input file \"%s\"\n", argv[2]);
        return EXIT_FAILURE;
    }

    const double t_io_read_start = hpc_gettime();
    //Initialize n_dims (D), n_points (N), *data with the datapoints in the input file
    read_input(inputf);
    fclose(inputf);
    const double t_io_read = hpc_gettime() - t_io_read_start;

    assert(n_clusters < n_points);

    if ((outputf = fopen(argv[3], "w")) == NULL) {
        fprintf(stderr, "FATAL: can not create output file \"%s\"\n", argv[3]);
        return EXIT_FAILURE;
    }

    fprintf(outputf, "# Data points: %d\n",     n_points);
    fprintf(outputf, "# Dimensions: %d\n",      n_dims);
    fprintf(outputf, "# Clusters: %d\n",        n_clusters);

    printf("\nInput file....... %s\n", argv[2]);
    printf("Output file...... %s\n", argv[3]);
    printf("Data points (N).. %d\n", n_points);
    printf("Dimensions (D)... %d\n", n_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);

    // allocate the remanining arrays. These are utility arrays that will be used and populated repeatedly during the
    // iterations of the algorithm
    centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));
    counts = (int*)safe_malloc(n_clusters * sizeof(*counts));

    //Populate *centroids vectors with random points that will be used as the initial centroids for the first iteration
    init_centroids();

    //printf("Main loop starts\n\n");

    float shift;
    int iter = 0;       //Keep track of how many iterations you have performed to eventually trigger the stopping condition
    
    double t_classify = 0.0;
    double t_update = 0.0;
    double t0 = 0.0;

    const double tstart = hpc_gettime();
    do {                        // 1) Inside the body of the main loop, you first classify each point according to the current centroids
        t0 = hpc_gettime();
        classify();
        t_classify += (hpc_gettime()-t0);


        t0 = hpc_gettime(); 
        shift = update_centroids();         //2) Then, update centroids according to the current classification
        t_update += (hpc_gettime()-t0);

        //printf("Iteration %3d, shift = %f\n", iter, shift);
        iter++;
    } while ( /*(shift > TOL) && */ (iter <= MAXITER) );     //STOPPING CONDITION: for perfomance measurements purposes, we set a fixed number of iterations (MAXITER)
    const double elapsed = hpc_gettime() - tstart;

    printf("\nMain loop completed\n");
    printf("Elapsed time %.3f\n\n", elapsed);

    const double t_io_write_start = hpc_gettime();
    save_results(outputf);
    fclose(outputf);
    const double t_io_write = hpc_gettime() - t_io_write_start;

    // Print results&stats
    printf("--- Clustering Statistics ---\n");
    printf("Input file       : %s\n", argv[2]);
    printf("Data points (N)  : %d\n", n_points);
    printf("Dimensions (D)   : %d\n", n_dims);
    printf("Clusters (K)     : %d\n", n_clusters);
    printf("Output file      : %s\n", argv[3]);
    printf("----- Time Merics - OpenMP -----\n");
    printf("t_io_read               : %f s\n", t_io_read);
    printf("LOOP - Total loop time  : %f s\n", elapsed);
    printf("LOOP - Classify time    : %f s\n", t_classify);
    printf("LOOP - Update time      : %f s\n", t_update);
    printf("t_io_write              : %f s\n", t_io_write);
    printf("--------------------------------\n\n");

    free(data);
    free(centroids);
    free(cluster_of);
    free(counts);

    return EXIT_SUCCESS;
}
