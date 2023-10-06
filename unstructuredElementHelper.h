#ifndef UNSTRUCTURED_ELEMENTS_HXX
#define UNSTRUCTURED_ELEMENTS_HXX

#include "owl/common.h"
// Note, the above dependency can be found here: https://github.com/owl-project/owl

namespace dtracker {
    using namespace owl;

    // -------------------------------------------------------------------------------------------------------------------------------
    // Unstructured element interpolation routines
    // Each element type has a corresponding interpolate<ELEMENT_TYPE> routine.
    // If the point lies within the the element, the interpolate routines return true, and false otherwise.
    // Per-vertex interpolated value is returned by reference.
    // -------------------------------------------------------------------------------------------------------------------------------

    #ifdef __CUDACC__
    #ifndef CUDA_DECORATOR
    #define CUDA_DECORATOR __both__
    #endif
    #else
    #ifndef CUDA_DECORATOR
    #define CUDA_DECORATOR
    #endif
    #endif

    // -------------------------------------------------------------------------------------------------------------------------------
    // TETRAHEDRA TYPE
    // -------------------------------------------------------------------------------------------------------------------------------

    /* computes the (oriented) volume of the tet given by the four vertices */
    inline CUDA_DECORATOR float
    volume(
        const vec3f &P,
        const vec3f &A,
        const vec3f &B,
        const vec3f &C)
    {
        return dot(P - A, cross(B - A, C - A));
    }

    inline CUDA_DECORATOR bool
    interpolateTetrahedra(
        const vec3f &P,
        const vec3f P0,
        const vec3f P1,
        const vec3f P2,
        const vec3f P3,
        const float S0,
        const float S1,
        const float S2,
        const float S3,
        float &value)
    {
        const float vol_all = volume(/*point*/ P0, /*base-tri*/ P1, P3, P2);
        if (vol_all == 0.f)
            return false;

        const float bary0 = volume(/*point*/ P, /*base-tri*/ P1, P3, P2) / vol_all;
        if (bary0 < 0.f)
            return false;

        const float bary1 = volume(/*point*/ P, /*base-tri*/ P0, P2, P3) / vol_all;
        if (bary1 < 0.f)
            return false;

        const float bary2 = volume(/*point*/ P, /*base-tri*/ P0, P3, P1) / vol_all;
        if (bary2 < 0.f)
            return false;

        const float bary3 = volume(/*point*/ P, /*base-tri*/ P0, P1, P2) / vol_all;
        if (bary3 < 0.f)
            return false;

    // If we're running the point sampling benchmark at this point we just
    // return true and skip doing the interpolation
    #ifdef POINT_QUERY_BENCHMARK
        return true;
    #else
        // attributes stored in w component of vertices
        value = (bary0 * S0 +
                 bary1 * S1 +
                 bary2 * S2 +
                 bary3 * S3);
        return true;
    #endif
    }

    // -------------------------------------------------------------------------------------------------------------------------------
    // PYRAMID TYPE
    // -------------------------------------------------------------------------------------------------------------------------------

    inline CUDA_DECORATOR void
    pyramidInterpolationFunctions(
        float pcoords[3],
        float sf[5])
    {
        float rm, sm, tm;

        rm = 1.f - pcoords[0];
        sm = 1.f - pcoords[1];
        tm = 1.f - pcoords[2];

        sf[0] = rm * sm * tm;
        sf[1] = pcoords[0] * sm * tm;
        sf[2] = pcoords[0] * pcoords[1] * tm;
        sf[3] = rm * pcoords[1] * tm;
        sf[4] = pcoords[2];
    }

    inline CUDA_DECORATOR void
    pyramidInterpolationDerivs(
        float pcoords[3],
        float derivs[15])
    {
        // r-derivatives
        derivs[0] = -(pcoords[1] - 1.f) * (pcoords[2] - 1.f);
        derivs[1] = (pcoords[1] - 1.f) * (pcoords[2] - 1.f);
        derivs[2] = pcoords[1] - pcoords[1] * pcoords[2];
        derivs[3] = pcoords[1] * (pcoords[2] - 1.f);
        derivs[4] = 0.f;

        // s-derivatives
        derivs[5] = -(pcoords[0] - 1.f) * (pcoords[2] - 1.f);
        derivs[6] = pcoords[0] * (pcoords[2] - 1.f);
        derivs[7] = pcoords[0] - pcoords[0] * pcoords[2];
        derivs[8] = (pcoords[0] - 1.f) * (pcoords[2] - 1.f);
        derivs[9] = 0.f;

        // t-derivatives
        derivs[10] = -(pcoords[0] - 1.f) * (pcoords[1] - 1.f);
        derivs[11] = pcoords[0] * (pcoords[1] - 1.f);
        derivs[12] = -pcoords[0] * pcoords[1];
        derivs[13] = (pcoords[0] - 1.f) * pcoords[1];
        derivs[14] = 1.f;
    }

    #define PYRAMID_DIVERGED 1.e6f
    #define PYRAMID_MAX_ITERATION 10
    #define PYRAMID_CONVERGED 1.e-04f
    #define PYRAMID_OUTSIDE_CELL_TOLERANCE 1.e-06f

    inline CUDA_DECORATOR bool
    interpolatePyramid(
        const vec3f &P,
        const vec3f P0,
        const vec3f P1,
        const vec3f P2,
        const vec3f P3,
        const vec3f P4,
        const float S0,
        const float S1,
        const float S2,
        const float S3,
        const float S4,
        float &value)
    {
        float params[3] = {0.5, 0.5, 0.5};
        float pcoords[3] = {0.5, 0.5, 0.5};
        float derivs[15];
        float weights[5];

        vec3f vertices[5] = {P0, P1, P2, P3, P4};

        memset(derivs, 0, sizeof(derivs));
        memset(weights, 0, sizeof(weights));

        const int edges[8][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}};

        float longestEdge = 0;
        for (int i = 0; i < 8; i++)
        {
            vec3f p0 = vertices[edges[i][0]];
            vec3f p1 = vertices[edges[i][1]];

            float dist = length(p1 - p0);
            if (longestEdge < dist)
                longestEdge = dist;
        }
        float volumeBound = powf(longestEdge, 3);
        float determinantTolerance =
            1e-20f < .00001f * volumeBound ? 1e-20f : .00001f * volumeBound;

        //  Enter iteration loop
        bool converged = false;
        for (int iteration = 0; !converged && (iteration < PYRAMID_MAX_ITERATION); iteration++)
        {
            //  calculate element interpolation functions and derivatives
            pyramidInterpolationFunctions(pcoords, weights);
            pyramidInterpolationDerivs(pcoords, derivs);
            //  calculate newton functions
            vec3f fcol = vec3f(0.f, 0.f, 0.f);
            vec3f rcol = vec3f(0.f, 0.f, 0.f);
            vec3f scol = vec3f(0.f, 0.f, 0.f);
            vec3f tcol = vec3f(0.f, 0.f, 0.f);
            for (int i = 0; i < 5; i++)
            {
                vec3f pt = (vec3f)vertices[i];
                fcol = fcol + pt * weights[i];
                rcol = rcol + pt * derivs[i];
                scol = scol + pt * derivs[i + 5];
                tcol = tcol + pt * derivs[i + 10];
            }
            fcol = fcol - P;
            // compute determinants and generate improvements
            float d = LinearSpace3f(rcol, scol, tcol).det();
            if (fabs(d) < determinantTolerance)
            {
                return false;
            }
            pcoords[0] = params[0] - LinearSpace3f(fcol, scol, tcol).det() / d;
            pcoords[1] = params[1] - LinearSpace3f(rcol, fcol, tcol).det() / d;
            pcoords[2] = params[2] - LinearSpace3f(rcol, scol, fcol).det() / d;
            // convergence/divergence test - if neither, repeat
            if (((fabs(pcoords[0] - params[0])) < PYRAMID_CONVERGED) &&
                ((fabs(pcoords[1] - params[1])) < PYRAMID_CONVERGED) &&
                ((fabs(pcoords[2] - params[2])) < PYRAMID_CONVERGED))
            {
                converged = true;
            }
            else if ((fabs(pcoords[0]) > PYRAMID_DIVERGED) ||
                    (fabs(pcoords[1]) > PYRAMID_DIVERGED) ||
                    (fabs(pcoords[2]) > PYRAMID_DIVERGED))
            {
                return false;
            }
            else
            {
                params[0] = pcoords[0];
                params[1] = pcoords[1];
                params[2] = pcoords[2];
            }
        }
        if (!converged)
        {
            return false;
        }
        float attrs[5];
        attrs[0] = S0;
        attrs[1] = S1;
        attrs[2] = S2;
        attrs[3] = S3;
        attrs[4] = S4;

        float lowerlimit = 0.0f - PYRAMID_OUTSIDE_CELL_TOLERANCE;
        float upperlimit = 1.0f + PYRAMID_OUTSIDE_CELL_TOLERANCE;
        if (pcoords[0] >= lowerlimit && pcoords[0] <= upperlimit &&
            pcoords[1] >= lowerlimit && pcoords[1] <= upperlimit &&
            pcoords[2] >= lowerlimit && pcoords[2] <= upperlimit)
        {
            // evaluation
            value = 0.f;
            pyramidInterpolationFunctions(pcoords, weights);
            for (int i = 0; i < 5; i++)
            {
                value += weights[i] * attrs[i];
            }
            return true;
        }
        return false;
    }

    // -------------------------------------------------------------------------------------------------------------------------------
    // WEDGE TYPE
    // -------------------------------------------------------------------------------------------------------------------------------
    inline CUDA_DECORATOR void
    wedgeInterpolationFunctions(
        float pcoords[3],
        float sf[6])
    {
        sf[0] = (1.0f - pcoords[0] - pcoords[1]) * (1.0f - pcoords[2]);
        sf[1] = pcoords[0] * (1.0f - pcoords[2]);
        sf[2] = pcoords[1] * (1.0f - pcoords[2]);
        sf[3] = (1.0f - pcoords[0] - pcoords[1]) * pcoords[2];
        sf[4] = pcoords[0] * pcoords[2];
        sf[5] = pcoords[1] * pcoords[2];
    }

    inline CUDA_DECORATOR void
    wedgeInterpolationDerivs(
        float pcoords[3],
        float derivs[18])
    {
        // r-derivatives
        derivs[0] = -1.0f + pcoords[2];
        derivs[1] = 1.0f - pcoords[2];
        derivs[2] = 0.0;
        derivs[3] = -pcoords[2];
        derivs[4] = pcoords[2];
        derivs[5] = 0.0;

        // s-derivatives
        derivs[6] = -1.0f + pcoords[2];
        derivs[7] = 0.0f;
        derivs[8] = 1.0f - pcoords[2];
        derivs[9] = -pcoords[2];
        derivs[10] = 0.0f;
        derivs[11] = pcoords[2];

        // t-derivatives
        derivs[12] = -1.0f + pcoords[0] + pcoords[1];
        derivs[13] = -pcoords[0];
        derivs[14] = -pcoords[1];
        derivs[15] = 1.0f - pcoords[0] - pcoords[1];
        derivs[16] = pcoords[0];
        derivs[17] = pcoords[1];
    }

    #define WEDGE_DIVERGED 1.e6f
    #define WEDGE_MAX_ITERATION 10
    #define WEDGE_CONVERGED 1.e-05f
    #define WEDGE_OUTSIDE_CELL_TOLERANCE 1.e-06f

    inline CUDA_DECORATOR bool
    interpolateWedge(
        const vec3f &P,
        const vec3f P0,
        const vec3f P1,
        const vec3f P2,
        const vec3f P3,
        const vec3f P4,
        const vec3f P5,
        const float S0,
        const float S1,
        const float S2,
        const float S3,
        const float S4,
        const float S5,
        float &value)
    {
        float params[3] = {0.5f, 0.5f, 0.5f};
        float pcoords[3] = {0.5f, 0.5f, 0.5f};
        float derivs[18];
        float weights[6];

        vec3f vertices[6] = {P0, P1, P2, P3, P4, P5};

        memset(derivs, 0, sizeof(derivs));
        memset(weights, 0, sizeof(weights));

        const int edges[9][2] = {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}};
        float longestEdge = 0;
        for (int i = 0; i < 9; i++)
        {
            vec3f p0 = vertices[edges[i][0]];
            vec3f p1 = vertices[edges[i][1]];

            float dist = length(p1 - p0);
            if (longestEdge < dist)
                longestEdge = dist;
        }

        float volumeBound = powf(longestEdge, 3);
        float determinantTolerance =
            1e-20f < .00001f * volumeBound ? 1e-20f : .00001f * volumeBound;

        //  enter iteration loop
        bool converged = false;
        for (int iteration = 0; !converged && (iteration < WEDGE_MAX_ITERATION); iteration++)
        {
            //  calculate element interpolation functions and derivatives
            wedgeInterpolationFunctions(pcoords, weights);
            wedgeInterpolationDerivs(pcoords, derivs);

            //  calculate newton functions
            vec3f fcol = vec3f(0.f, 0.f, 0.f);
            vec3f rcol = vec3f(0.f, 0.f, 0.f);
            vec3f scol = vec3f(0.f, 0.f, 0.f);
            vec3f tcol = vec3f(0.f, 0.f, 0.f);
            for (int i = 0; i < 6; i++)
            {
                vec3f pt = vertices[i];
                fcol = fcol + pt * weights[i];
                rcol = rcol + pt * derivs[i];
                scol = scol + pt * derivs[i + 6];
                tcol = tcol + pt * derivs[i + 12];
            }

            fcol = fcol - P;

            // compute determinants and generate improvements
            float d = LinearSpace3f(rcol, scol, tcol).det();
            if (fabs(d) < determinantTolerance)
            {
                return false;
            }
            pcoords[0] = params[0] - LinearSpace3f(fcol, scol, tcol).det() / d;
            pcoords[1] = params[1] - LinearSpace3f(rcol, fcol, tcol).det() / d;
            pcoords[2] = params[2] - LinearSpace3f(rcol, scol, fcol).det() / d;

            // convergence/divergence test - if neither, repeat
            if (((fabs(pcoords[0] - params[0])) < WEDGE_CONVERGED) &&
                ((fabs(pcoords[1] - params[1])) < WEDGE_CONVERGED) &&
                ((fabs(pcoords[2] - params[2])) < WEDGE_CONVERGED))
            {
                converged = true;
            }
            else if ((fabs(pcoords[0]) > WEDGE_DIVERGED) ||
                    (fabs(pcoords[1]) > WEDGE_DIVERGED) ||
                    (fabs(pcoords[2]) > WEDGE_DIVERGED))
            {
                return false;
            }
            else
            {
                params[0] = pcoords[0];
                params[1] = pcoords[1];
                params[2] = pcoords[2];
            }
        }

        if (!converged)
        {
            return false;
        }

        float attrs[6];
        attrs[0] = S0;
        attrs[1] = S1;
        attrs[2] = S2;
        attrs[3] = S3;
        attrs[4] = S4;
        attrs[5] = S5;

        float lowerlimit = 0.0f - WEDGE_OUTSIDE_CELL_TOLERANCE;
        float upperlimit = 1.0f + WEDGE_OUTSIDE_CELL_TOLERANCE;
        if (pcoords[0] >= lowerlimit && pcoords[0] <= upperlimit &&
            pcoords[1] >= lowerlimit && pcoords[1] <= upperlimit &&
            pcoords[2] >= lowerlimit && pcoords[2] <= upperlimit &&
            pcoords[0] + pcoords[1] <= upperlimit)
        {
            // evaluation
            value = 0.f;
            wedgeInterpolationFunctions(pcoords, weights);
            for (int i = 0; i < 6; i++)
            {
                value += weights[i] * attrs[i];
            }

            // value = .5;
            return true;
        }
        return false;
    }

    // -------------------------------------------------------------------------------------------------------------------------------
    // HEXAHEDRA TYPE
    // -------------------------------------------------------------------------------------------------------------------------------
    inline CUDA_DECORATOR void
    hexInterpolationFunctions(
        float pcoords[3],
        float sf[8])
    {
        float rm, sm, tm;

        rm = 1.f - pcoords[0];
        sm = 1.f - pcoords[1];
        tm = 1.f - pcoords[2];

        sf[0] = rm * sm * tm;
        sf[1] = pcoords[0] * sm * tm;
        sf[2] = pcoords[0] * pcoords[1] * tm;
        sf[3] = rm * pcoords[1] * tm;
        sf[4] = rm * sm * pcoords[2];
        sf[5] = pcoords[0] * sm * pcoords[2];
        sf[6] = pcoords[0] * pcoords[1] * pcoords[2];
        sf[7] = rm * pcoords[1] * pcoords[2];
    }

    inline CUDA_DECORATOR void
    hexInterpolationDerivs(
        float pcoords[3],
        float derivs[24])
    {
        float rm, sm, tm;

        rm = 1.f - pcoords[0];
        sm = 1.f - pcoords[1];
        tm = 1.f - pcoords[2];

        // r-derivatives
        derivs[0] = -sm * tm;
        derivs[1] = sm * tm;
        derivs[2] = pcoords[1] * tm;
        derivs[3] = -pcoords[1] * tm;
        derivs[4] = -sm * pcoords[2];
        derivs[5] = sm * pcoords[2];
        derivs[6] = pcoords[1] * pcoords[2];
        derivs[7] = -pcoords[1] * pcoords[2];

        // s-derivatives
        derivs[8] = -rm * tm;
        derivs[9] = -pcoords[0] * tm;
        derivs[10] = pcoords[0] * tm;
        derivs[11] = rm * tm;
        derivs[12] = -rm * pcoords[2];
        derivs[13] = -pcoords[0] * pcoords[2];
        derivs[14] = pcoords[0] * pcoords[2];
        derivs[15] = rm * pcoords[2];

        // t-derivatives
        derivs[16] = -rm * sm;
        derivs[17] = -pcoords[0] * sm;
        derivs[18] = -pcoords[0] * pcoords[1];
        derivs[19] = -rm * pcoords[1];
        derivs[20] = rm * sm;
        derivs[21] = pcoords[0] * sm;
        derivs[22] = pcoords[0] * pcoords[1];
        derivs[23] = rm * pcoords[1];
    }

    #define HEX_DIVERGED 1.e6f
    #define HEX_MAX_ITERATION 10
    #define HEX_CONVERGED 1.e-05f
    #define HEX_OUTSIDE_CELL_TOLERANCE 1.e-06f

    inline CUDA_DECORATOR bool
    interpolateHexahedra(
        const vec3f &P,
        const vec3f P0,
        const vec3f P1,
        const vec3f P2,
        const vec3f P3,
        const vec3f P4,
        const vec3f P5,
        const vec3f P6,
        const vec3f P7,
        const float S0,
        const float S1,
        const float S2,
        const float S3,
        const float S4,
        const float S5,
        const float S6,
        const float S7,
        float &value)
    {
        float params[3] = {0.5, 0.5, 0.5};
        float pcoords[3];
        float derivs[24];
        float weights[8];

        vec3f vertices[8] = {P0, P1, P2, P3, P4, P5, P6, P7};

        // Should precompute these
        const int diagonals[4][2] = {{0, 6}, {1, 7}, {2, 4}, {3, 5}};
        float longestDiagonal = 0;
        for (int i = 0; i < 4; i++)
        {
            const vec3f p0 = vertices[diagonals[i][0]];
            const vec3f p1 = vertices[diagonals[i][1]];
            float dist = length(p1 - p0);
            if (longestDiagonal < dist)
                longestDiagonal = dist;
        }

        const float volumeBound = longestDiagonal * longestDiagonal * longestDiagonal;
        const float determinantTolerance =
            1e-20f < .00001f * volumeBound ? 1e-20f : .00001f * volumeBound;

        // Set initial position for Newton's method
        pcoords[0] = pcoords[1] = pcoords[2] = 0.5;

        // Enter iteration loop
        bool converged = false;
        for (int iteration = 0; !converged && (iteration < HEX_MAX_ITERATION); iteration++)
        {
            // Calculate element interpolation functions and derivatives
            hexInterpolationFunctions(pcoords, weights);
            hexInterpolationDerivs(pcoords, derivs);

            // Calculate newton functions
            vec3f fcol = vec3f(0.f, 0.f, 0.f);
            vec3f rcol = vec3f(0.f, 0.f, 0.f);
            vec3f scol = vec3f(0.f, 0.f, 0.f);
            vec3f tcol = vec3f(0.f, 0.f, 0.f);
            for (int i = 0; i < 8; i++)
            {
                const vec3f pt = vertices[i];
                fcol = fcol + pt * weights[i];
                rcol = rcol + pt * derivs[i];
                scol = scol + pt * derivs[i + 8];
                tcol = tcol + pt * derivs[i + 16];
            }

            fcol = fcol - P;

            // Compute determinants and generate improvements
            float d = LinearSpace3f(rcol, scol, tcol).det();
            if (fabs(d) < determinantTolerance)
            {
                return false;
            }

            pcoords[0] = params[0] - LinearSpace3f(fcol, scol, tcol).det() / d;
            pcoords[1] = params[1] - LinearSpace3f(rcol, fcol, tcol).det() / d;
            pcoords[2] = params[2] - LinearSpace3f(rcol, scol, fcol).det() / d;

            // Convergence/divergence test - if neither, repeat
            if (((fabs(pcoords[0] - params[0])) < HEX_CONVERGED) &&
                ((fabs(pcoords[1] - params[1])) < HEX_CONVERGED) &&
                ((fabs(pcoords[2] - params[2])) < HEX_CONVERGED))
            {
                converged = true;
            }
            else if ((fabs(pcoords[0]) > HEX_DIVERGED) ||
                    (fabs(pcoords[1]) > HEX_DIVERGED) ||
                    (fabs(pcoords[2]) > HEX_DIVERGED))
            {
                return false;
            }
            else
            {
                params[0] = pcoords[0];
                params[1] = pcoords[1];
                params[2] = pcoords[2];
            }
        }

        if (!converged)
        {
            return false;
        }

        float attrs[8];
        attrs[0] = S0;
        attrs[1] = S1;
        attrs[2] = S2;
        attrs[3] = S3;
        attrs[4] = S4;
        attrs[5] = S5;
        attrs[6] = S6;
        attrs[7] = S7;

        const float lowerlimit = 0.0f - HEX_OUTSIDE_CELL_TOLERANCE;
        const float upperlimit = 1.0f + HEX_OUTSIDE_CELL_TOLERANCE;
        if ((pcoords[0] >= lowerlimit && pcoords[0] <= upperlimit &&
            pcoords[1] >= lowerlimit && pcoords[1] <= upperlimit &&
            pcoords[2] >= lowerlimit && pcoords[2] <= upperlimit))
        {
            // Evaluation
            value = 0.f;
            hexInterpolationFunctions(pcoords, weights);
            for (int i = 0; i < 8; i++)
            {
                value += weights[i] * attrs[i];
            }

            return true;
        }
        return false;
    }
}

#endif