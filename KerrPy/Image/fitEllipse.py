# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2014-05-07 JMT From scratch
REFS   : Based on the robust pupil tracker developed in Swirski et al, 2012
This file is part of mrgaze.
   mrgaze is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   mrgaze is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU General Public License
   along with mrgaze.  If not, see <http://www.gnu.org/licenses/>.
Copyright 2014 California Institute of Technology.
"""

import numpy as np
import cv2

#---------------------------------------------
# Ellipse Fitting Functions
#---------------------------------------------


def FitEllipse_LeastSquares(pnts):
    '''
    Simple least-squares ellipse fit to boundary points
    Parameters
    ----
    pnts : n x 2 array of integers
        Candidate pupil-iris boundary points from edge detection
    roi : 2D scalar array
        Grayscale image of pupil-iris region for display only
    cfg : configuration structure
        Configuration parameters
    Returns
    ----
    best_ellipse : tuple of tuples
        Best fitted ellipse parameters ((x0, y0), (a,b), theta)
    '''

    # Tiny circle init
    best_ellipse = ((0,0),(1e-6,1e-6),0)

    # Break if too few points to fit ellipse (RARE)
    if pnts.shape[0] < 5:
        return best_ellipse

    # Call OpenCV ellipse fitting
    best_ellipse = cv2.fitEllipse(pnts)
    
    return best_ellipse


def EllipseError(pnts, ellipse):
    """
    Ellipse fit error function
    """

    # Suppress divide-by-zero warnings
    np.seterr(divide='ignore')

    # Calculate algebraic distances and gradients of all points from fitted ellipse
    distance, grad, absgrad, normgrad = ConicFunctions(pnts, ellipse)

    # Calculate error from distance and gradient
    # See Swirski et al 2012
    # TODO : May have to use distance / |grad|^0.45 - see Swirski source

    # Gradient array has x and y components in rows (see ConicFunctions)
    err = distance / absgrad

    return err


def EllipseNormError(pnts, ellipse):
    """
    Error normalization factor, alpha
    Normalizes cost to 1.0 at point 1 pixel out from minor vertex along minor axis
    """

    # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
    # Where aa and bb are the major and minor axes, and phi_b_deg
    # is the CW x to minor axis rotation in degrees
    (x0,y0), (bb,aa), phi_b_deg = ellipse

    # Semiminor axis
    b = bb/2

    # Convert phi_b from deg to rad
    phi_b_rad = phi_b_deg * np.pi / 180.0

    # Minor axis vector
    bx, by = np.cos(phi_b_rad), np.sin(phi_b_rad)

    # Point one pixel out from ellipse on minor axis
    p1 = np.array( (x0 + (b + 1) * bx, y0 + (b + 1) * by) ).reshape(1,2)

    # Error at this point
    err_p1 = EllipseError(p1, ellipse)

    # Errors at provided points
    err_pnts = EllipseError(pnts, ellipse)

    return err_pnts / err_p1


def EllipseSupport(pnts, ellipse, dIdx, dIdy):
    """
    Ellipse support function
    """

    if pnts.size < 5:
        return -np.inf

    # Return sum of (grad Q . grad image) over point set
    return EllipseImageGradDot(pnts, ellipse, dIdx, dIdy).sum()


def EllipseImageGradDot(pnts, ellipse, dIdx, dIdy):

    # Calculate normalized grad Q at inlier pnts
    distance, grad, absgrad, normgrad = ConicFunctions(pnts, ellipse)

    # Extract vectors of x and y values
    x, y = pnts[:,0], pnts[:,1]

    # Extract image gradient at inlier points
    dIdx_pnts = dIdx[y,x]
    dIdy_pnts = dIdy[y,x]

    # Construct intensity gradient array (2 x N)
    gradI = np.array( (dIdx_pnts, dIdy_pnts) )

    # Calculate the sum of the column-wise dot product of normgrad and gradI
    # http://stackoverflow.com/questions/6229519/numpy-column-wise-dot-product
    return np.einsum('ij,ij->j', normgrad, gradI)


#---------------------------------------------
# Ellipse Math
#---------------------------------------------


def Geometric2Conic(ellipse):
    """
    Geometric to conic parameter conversion
    References
    ----
    Adapted from Swirski's ConicSection.h
    https://bitbucket.org/Leszek/pupil-tracker/
    """

    # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
    # Where aa and bb are the major and minor axes, and phi_b_deg
    # is the CW x to minor axis rotation in degrees
    (x0,y0), (bb, aa), phi_b_deg = ellipse

    # Semimajor and semiminor axes
    a, b = aa/2, bb/2

    # Convert phi_b from deg to rad
    phi_b_rad = phi_b_deg * np.pi / 180.0

    # Major axis unit vector
    ax, ay = -np.sin(phi_b_rad), np.cos(phi_b_rad)

    # Useful intermediates
    a2 = a*a
    b2 = b*b

    #
    # Conic parameters
    #
    if a2 > 0 and b2 > 0:

        A = ax*ax / a2 + ay*ay / b2;
        B = 2*ax*ay / a2 - 2*ax*ay / b2;
        C = ay*ay / a2 + ax*ax / b2;
        D = (-2*ax*ay*y0 - 2*ax*ax*x0) / a2 + (2*ax*ay*y0 - 2*ay*ay*x0) / b2;
        E = (-2*ax*ay*x0 - 2*ay*ay*y0) / a2 + (2*ax*ay*x0 - 2*ax*ax*y0) / b2;
        F = (2*ax*ay*x0*y0 + ax*ax*x0*x0 + ay*ay*y0*y0) / a2 + (-2*ax*ay*x0*y0 + ay*ay*x0*x0 + ax*ax*y0*y0) / b2 - 1;

    else:

        # Tiny dummy circle - response to a2 or b2 == 0 overflow warnings
        A,B,C,D,E,F = (1,0,1,0,0,-1e-6)

    # Compose conic parameter array
    conic = np.array((A,B,C,D,E,F))

    return conic


def Conic2Geometric(conic):
    """
    Merge geometric parameter functions from van Foreest code
    References
    ----
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """

    # Extract modified conic parameters
    A,B,C,D,E,F = conic[0], conic[1]/2, conic[2], conic[3]/2, conic[4]/2, conic[5]

    # Usefult intermediates
    dAC = A-C
    Z = np.sqrt( 1 + 4*B*B/(dAC*dAC) )

    # Center
    num = B * B - A * C
    x0 = (C * D - B * E) / num
    y0 = (A * E - B * D) / num

    # Axis lengths
    up    = 2 * (A*E*E + C*D*D + F*B*B - 2*B*D*E - A*C*F)
    down1 = (B*B-A*C) * ( -dAC*Z - (C+A) )
    down2 = (B*B-A*C) * (  dAC*Z - (C+A) )
    b, a  = np.sqrt(up/down1), np.sqrt(up/down2)

    # Minor axis rotation angle in degrees (CW from x axis, origin upper left)
    phi_b_deg =  0.5 * np.arctan(2 * B / dAC) * 180.0 / np.pi

    # Note OpenCV ellipse parameter format (full axes)
    return (x0,y0), (2*b, 2*a), phi_b_deg


def ConicFunctions(pnts, ellipse):
    """
    Calculate various conic quadratic curve support functions
    General 2D quadratic curve (biquadratic)
    Q = Ax^2 + Bxy + Cy^2 + Dx + Ey + F
    For point on ellipse, Q = 0, with appropriate coefficients
    Parameters
    ----
    pnts : n x 2 array of floats
    ellipse : tuple of tuples
    Returns
    ----
    distance : array of floats
    grad : array of floats
    absgrad : array of floats
    normgrad : array of floats
    References
    ----
    Adapted from Swirski's ConicSection.h
    https://bitbucket.org/Leszek/pupil-tracker/
    """

    # Suppress invalid values
    np.seterr(invalid='ignore')

    # Convert from geometric to conic ellipse parameters
    conic = Geometric2Conic(ellipse)

    # Row vector of conic parameters (Axx, Axy, Ayy, Ax, Ay, A1) (1 x 6)
    C = np.array(conic)

    # Extract vectors of x and y values
    x, y = pnts[:,0], pnts[:,1]

    # Construct polynomial array (6 x n)
    X = np.array( ( x*x, x*y, y*y, x, y, np.ones_like(x) ) )

    # Calculate Q/distance for all points (1 x n)
    distance = C.dot(X)

    # Quadratic curve gradient at (x,y)
    # Analytical grad of Q = Ax^2 + Bxy + Cy^2 + Dx + Ey + F
    # (dQ/dx, dQ/dy) = (2Ax + By + D, Bx + 2Cy + E)

    # Construct conic gradient coefficients vector (2 x 3)
    Cg = np.array( ( (2*C[0], C[1], C[3]), (C[1], 2*C[2], C[4]) ) )

    # Construct polynomial array (3 x n)
    Xg = np.array( (x, y, np.ones_like(x) ) )

    # Gradient array (2 x n)
    grad = Cg.dot(Xg)

    # Normalize gradient -> unit gradient vector
    # absgrad = np.apply_along_axis(np.linalg.norm, 0, grad)
    absgrad = np.sqrt(np.sqrt(grad[0,:]**2 + grad[1,:]**2))
    normgrad = grad / absgrad

    return distance, grad, absgrad, normgrad


def Eccentricity(ellipse):
    '''
    Calculate eccentricity of an ellipse
    '''

    # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
    # Where aa and bb are the major and minor axes, and phi_b_deg
    # is the CW x to minor axis rotation in degrees
    (x0,y0), (bb, aa), phi_b_deg = ellipse

    return np.sqrt(1 - (bb/aa)**2)


def OverlayRANSACFit(img, all_pnts, inlier_pnts, ellipse):
    """
    NOTE
    ----
    All points are (x,y) pairs, but arrays are (row, col) so swap
    coordinate ordering for correct positioning in array
    """
    
    img_overlay = img
    
    # linewidth for the Canny edge
    lw = 5
    # to avoid out-of-range index error, let us use Periodic Boundary Conditions
    img_width = img.shape[0]
    img_height = img.shape[1]
    
    
    # Overlay all pnts in red
    for col, row in all_pnts:
        for i in np.arange(lw):
            for j in np.arange(lw):
                row_periodic = (row + i)% img_width
                col_periodic = (col + j)% img_height
                img[row_periodic, col_periodic] = [0,0,255]

    # Overlay inliers in green
    for col, row in inlier_pnts:
        for i in np.arange(lw):
            for j in np.arange(lw):
                row_periodic = (row + i)% img_width
                col_periodic = (col + j)% img_height
                img[row_periodic, col_periodic] = [0,255,0]
    
    # Overlay inlier fitted ellipse in red
    # cv2.ellipse(img, ((ellipse[0][0],ellipse[0][1]),(ellipse[1][0],ellipse[1][1]),ellipse[2]), (255,0,0), 5)
    
    #rotate the ellipse by 90
    # rot_ellipse = ((ellipse[0][1],ellipse[0][0]),(ellipse[1][1],ellipse[1][0]),ellipse[2])

    img_overlay = cv2.ellipse(img_overlay, ellipse, (0, 255,255), 3)
    return img_overlay

def ConfidenceInFit(pnts, ellipse, max_norm_err_sq, DEBUG):
    """
    Given an ellipse and the points, return the percentage of points which lie within a given allowed error of the ellipse. This sets the confidence of the fit
    """
    # Count pnts (n x 2)
    n_pnts = pnts.shape[0]
    # Calculate normalized errors for all points
    norm_err = EllipseNormError(pnts, ellipse)

    # Identify inliers
    inliers = np.nonzero(norm_err**2 < max_norm_err_sq)[0]
    
    # Update inliers set
    inlier_pnts = pnts[inliers]
    
    # Count inliers (n x 2)
    n_inliers    = inliers.size

    perc_inliers = 0.0    
    
    try:
        
        perc_inliers = (n_inliers * 100.0) / n_pnts

    except:
        
        print(" ALERT! There are no edges...")
        perc_inliers = 0.0
        
    if DEBUG: print(f"                c(fit): {np.int(perc_inliers)}%")
    return perc_inliers, inlier_pnts, norm_err
    
if __name__ == '__main__':
    unittest = True
    DEBUG= True
    max_norm_err_sq =   20.0    # maximum allowed normalized error allowed for a point to be an inlier

#    img_edges = cv2.imread('06_CannyEdges.png', 0)
    
    
    #load the Canny edges
    edges = np.load('06_CannyEdges.npy')
    
    #load the median filtered image
    img_med = cv2.imread('03_MedianFiltered.png',0)
    
    #create a color image
#    img_color = cv2.merge((edges, edges, edges))
    img_color = cv2.merge((img_med, img_med, img_med))
    pnts = np.transpose(np.nonzero(edges))

    ellipse = FitEllipse_LeastSquares(pnts)
    perc_inliers, inlier_pnts, norm_err = ConfidenceInFit (pnts, ellipse, max_norm_err_sq, DEBUG)
    
    OverlayRANSACFit(img_color, pnts, inlier_pnts, ellipse)
