# comparing to feature_calc_5_simple_corr_edit.py
# adding correlation length for the center cells.

import pandas as pd
import numpy as np

# calculating circular correlation
import hoggorm as ho
from astropy.stats import circcorrcoef
from astropy import units as u

from scipy.stats import pearsonr
from scipy.interpolate import griddata
import scipy
from scipy.spatial import distance
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from skimage.morphology import square, dilation
from skimage import data, util
from skimage.measure import label, perimeter

import seaborn as sns
import os
import re
import json
import sys

import pipe_util2
import cc3d_util as cu

from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay, delaunay_plot_2d
from skimage.draw import polygon
from skimage.measure import label, regionprops, regionprops_table, perimeter
from scipy.stats import entropy

def normalize(v):
    """
    This function is called by read_time_series
    When calculating single cell polarity time series
    Return a normalized vector
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def read_time_series(start, end, interval, data_folder, rep_inx, iter_inx):
    
    """
    obtain a list of polarity related values from some time interval.
    start - int, starting timepoint
    end - int, stop timepoint
    interval - int
    data_folder - string
    rep_inx - int, index for repeat
    iter_inx - inx, index for parameter scan iteration
    """

    past_px_com = [] # record x positions
    past_py_com = [] # record y positions
    
    past_fx = []
    past_fy = []

    t_list = np.arange(start, end+1, interval, dtype = np.intc)
    for t in t_list:
        past_traj = cu.traj_file(data_folder, rep_inx, iter_inx, t)
        past_df = pd.read_csv(past_traj)
        
        # list values for all cells
        px_com = past_df.x # cell x position
        py_com = past_df.y # cell y position
        fx = -past_df.f_x # force applied on x axis
        fy = -past_df.f_y # force applied on y axis

        # each row is info for all cells
        # col are time
        past_px_com.append(px_com)
        past_py_com.append(py_com)
        past_fx.append(fx)
        past_fy.append(fy)
        
        
    # calc polarity with time intervel 5 MCS
    past_px_com = np.asarray(past_px_com)
    past_py_com = np.asarray(past_py_com)
    
    past_angle = []
    past_px = []
    past_py = []
    i = 1
    while i<len(past_px_com):

        px_i = past_px_com[i]-past_px_com[i-1] # velocity vector over specfied time intervel
        py_i = past_py_com[i]-past_py_com[i-1]
        
        pos_x = []
        pos_y = []
        # polarity pair for each cell
        j = 0
        while j<len(px_i):
            px_i_j = px_i[j]
            py_i_j = py_i[j]
            norms = normalize([px_i_j, py_i_j])
            pos_x.append(norms[0]) # polarity vector over specified intervel - x axis
            pos_y.append(norms[1]) # polarity vector over specified intervel - y axis
            
            j = j+1
            
        angle = np.arctan2(pos_x, pos_y)

        past_px.append(pos_x)
        past_py.append(pos_y)
        past_angle.append(angle)

        i = i+1

    # each row is time series for a cell
    past_px = np.array(past_px).T         # single cell polarities - x
    past_py = np.array(past_py).T         # single cell polarities - y
    past_fx = np.array(past_fx).T         # force excerted on cell - x
    past_fy = np.array(past_fy).T         # force excerted on cell - y
    past_angle = np.array(past_angle).T   # single cell polarity angle [-pi, pi]
    
    return (past_px, past_py, past_fx, past_fy, past_angle)

def read_time_series2(start, end, data_folder, rep_inx, iter_inx):
    
    """
    obtain a list of polarity related values from some time interval.
    start - int, starting timepoint
    end - int, stop timepoint
    interval - int
    data_folder - string
    rep_inx - int, index for repeat
    iter_inx - inx, index for parameter scan iteration
    """

    past_px_com = [] # record x positions
    past_py_com = [] # record y positions
    
    past_fx = []
    past_fy = []
    
    p_tot_list = [] # tumor polarity over time

    t_list = np.arange(start-5, end+1, 1, dtype = np.intc)
    for t in t_list:
        past_traj = cu.traj_file(data_folder, rep_inx, iter_inx, t, verbose = 0)
        past_df = pd.read_csv(past_traj)
        
        # list values for all cells
        px_com = past_df.x # cell x position
        py_com = past_df.y # cell y position
        fx = -past_df.f_x # force applied on x axis
        fy = -past_df.f_y # force applied on y axis

        # each row is info for all cells
        # col are time
        past_px_com.append(px_com)
        past_py_com.append(py_com)
        past_fx.append(fx)
        past_fy.append(fy)
        
        
    # calc polarity with time intervel 5 MCS
    past_px_com = np.asarray(past_px_com)
    past_py_com = np.asarray(past_py_com)
    
    past_angle = []
    past_px = []
    past_py = []
    i = start
    while i<len(past_px_com):

        px_i = past_px_com[i]-past_px_com[i-5] # velocity vector over specfied time intervel
        py_i = past_py_com[i]-past_py_com[i-5]
        
        pos_x = []
        pos_y = []
        # polarity pair for each cell
        j = 0
        while j<len(px_i):
            px_i_j = px_i[j]
            py_i_j = py_i[j]
            norms = normalize([px_i_j, py_i_j])
            pos_x.append(norms[0]) # polarity vector over specified intervel - x axis
            pos_y.append(norms[1]) # polarity vector over specified intervel - y axis
            
            j = j+1
            
        angle = np.arctan2(pos_x, pos_y)

        past_px.append(pos_x)
        past_py.append(pos_y)
        past_angle.append(angle)
        
        pos_x_sum = sum(pos_x)
        pos_y_sum = sum(pos_y)
        pos_tot = (pos_x_sum**2+pos_y_sum**2)**0.5/len(pos_x)
        p_tot_list.append(pos_tot)
        

        i = i+1

    # each row is time series for a cell
    past_px = np.array(past_px).T         # single cell polarities - x
    past_py = np.array(past_py).T         # single cell polarities - y
    past_fx = np.array(past_fx).T         # force excerted on cell - x
    past_fy = np.array(past_fy).T         # force excerted on cell - y
    past_angle = np.array(past_angle).T   # single cell polarity angle [-pi, pi]
    
    return (past_px, past_py, past_fx, past_fy, past_angle, p_tot_list)

def polarity_mean(past_px, past_py):
    
    """
    Calculate polarity
    """
    
#     print("past_px: ", past_px.shape)
    
    # calc tumor polarity for each timepoint 
    tumor_px_sum = past_px.sum(axis = 0, keepdims=True)
    tumor_py_sum = past_py.sum(axis = 0, keepdims=True)
    tumor_p_norm = (tumor_px_sum**2+tumor_py_sum**2)**0.5/len(past_px)
    tumor_p_mean = np.mean(tumor_p_norm)
#     print("tumor_px_sum shape: ", tumor_px_sum.shape, tumor_px_sum)
#     print("tumor_p_mean", tumor_p_mean)
    
    # average polarity for a single cell in the timecourse
    px_time_mean = past_px.mean(axis = 1, keepdims=True)
    py_time_mean = past_py.mean(axis = 1, keepdims=True)
#     print("px_time_mean", px_time_mean.shape)
    # variance 2, variance over time then average for each cell.
    cell_p_var = np.mean(((past_px-px_time_mean)**2+(past_py-py_time_mean)**2)**0.5, axis=1).mean()

    return (tumor_p_mean, cell_p_var)

def perimeter_coords(array):
    
    x,y = np.nonzero(array)
    #print(x)
    peri_x = []
    peri_y = []
    i = 0
    while i<len(x):
        x_inx_min = x[i]-1
        x_inx_max = x[i]+2
        y_inx_min = y[i]-1
        y_inx_max = y[i]+2
        
        if x[i]==0:
            x_inx_min = 0
        if y[i]==0:
            y_inx_min = 0
        
        #print(x_inx_min, x_inx_max, y_inx_min, y_inx_max)
        part = array[x_inx_min:x_inx_max, y_inx_min:y_inx_max]
        size = part.shape[0]*part.shape[1]
        local = np.sum(part)
        
        if local<size:
            # this is perimeter point
            peri_x.append(x[i])
            peri_y.append(y[i])
        
        i = i+1
    
    peri_x = np.array(peri_x)
    peri_y = np.array(peri_y)

    return peri_x, peri_y


def tumor_geo_calc(data_folder, feature_folder, figure_folder,
                   rep_inx, iter_inx, timepoint):
    
    data_folder = pipe_util2.folder_verify(data_folder)
    feature_foler = pipe_util2.folder_verify(feature_folder)
    figure_folder = pipe_util2.folder_verify(figure_folder)
    CHECK_CSV_FOLDER = os.path.isdir(feature_folder)
    CHECK_PNG_FOLDER = os.path.isdir(figure_folder)
    if not CHECK_CSV_FOLDER:
        pipe_util2.create_folder(feature_folder)
    if not CHECK_PNG_FOLDER:
        pipe_util2.create_folder(figure_folder)
    
    "Scan parameters"
    # obtain the real scan index
    scan_f = cu.scan_iter_dirs(data_folder, rep_inx, iter_inx)
    scan_f_name = os.path.basename(scan_f[:-1])
    print("haha", scan_f, scan_f_name)
    scan_inx = scan_f_name.split("_")[2]
    scan_inx = int(scan_inx)
    
    jasn_file = pipe_util2.folder_file_num(scan_f, ".json")[0]
    with open(jasn_file) as f:
        data = json.load(f)
    #print(data["parameters"].keys())
    scan_keys = list(data["parameters"].keys())
    scan_param = []
    for key in scan_keys:
        scan_param.append(float(data["parameters"][key]))
    scan_keys = " ".join(scan_keys)
    scan_param = " ".join(np.array(scan_param).astype(str))
    #print(scan_keys, scan_param)
    
    traj = cu.traj_file(data_folder, rep_inx, iter_inx, timepoint)
    df = pd.read_csv(traj)
    
    "Intrinsic profile"
    rac_max = df["rac"].max()
    rac_mea = df["rac"].mean()
    # tumor center displacement from center
    s = ((df.x.mean()-128)**2+(df.y.mean()-128)**2)**0.5
    # average single cell displacment from center
    d = (((df.x-128)**2+(df.y-128)**2)**0.5).mean()
    # average single cell displacement from center std
    d_std = (((df.x-128)**2+(df.y-128)**2)**0.5).std()
    
    aa = df.a.max()
    ss = df.s.max()
    f_coef = df.f_coef.max()

    force_max = df.f.max() # average force for the tumor
    force_mea = df.f.mean()
    fpp = df.fpp.mean() # average fpp 
    p_frac = df.p_frac.max()

    px = df.x_self_polarity
    py = df.y_self_polarity
    ang = np.arctan2(py, px)
    ang_std = np.std(ang)
    ang_mea = np.mean(ang)

    # polarity
    top = (px.sum()**2+py.sum()**2)**0.5
    bot = ((px**2+py**2)**0.5).sum()
    #print(top)
    if bot==0:
        polarity = 0.
    else:
        polarity = top/bot
        
    
    # ------------- calc correlation length ----------------
    # locating the center cell
    # at 90 mcs, add secretome
    traj_90 = cu.traj_file(data_folder, rep_inx, iter_inx, 90)
    df_90 = pd.read_csv(traj_90)
    c_cell_inx = ((df_90.x-128)**2+(df_90.y-128)**2).idxmin()
    #print(c_cell_inx)
    
    x_c = df.x[c_cell_inx] # center cell x
    y_c = df.y[c_cell_inx] # center cell y
    
    traj_minus_1 = cu.traj_file(data_folder, rep_inx, iter_inx, timepoint-5)
    df_minus_1 = pd.read_csv(traj_minus_1)
    
    vx = df.x-df_minus_1.x # velocity vector x
    vy = df.y-df_minus_1.y # velocity vector y
    
    vx_c = vx[c_cell_inx] # center cell velocity vector x
    vy_c = vy[c_cell_inx] # center cell velocity vector y
    
    name = "rep_"+"%s"%rep_inx+"_scan_"+"%s"%scan_inx+"_time_"+"%s"%timepoint
#     print(vx_c, vy_c)
    if vx_c ==0 and vy_c ==0: # in case center cell velocity is zero
        r_list = [0]
        corr_list = [0]
        correlation_length = 0
        
        plt.figure(figsize=(5, 3))
        plt.grid()
        plt.scatter(r_list, corr_list, label="Data", color='b', lw = 0, s = 22)
        plt.ylim((-1,1))
        plt.xlim((0,80))
        plt.xlabel("Distance $D_{toccell}$, pixel")
        plt.ylabel("Velocity Correlation")
        plt.legend(frameon = False, facecolor='white', framealpha=1, loc = (0.6,1))

        plt.savefig(figure_folder + name+ "_correlation.png", dpi = 200, bbox_inches = "tight")
    
    else:
        rs = ((df.x-x_c)**2+(df.y-y_c)**2)**0.5

        r_list = []
        corr_list = []
        dr = 1
        r = dr
        while r<128:

            vx_sub = vx[((rs>r-dr)&(rs<=r))]
            vy_sub = vy[((rs>r-dr)&(rs<=r))]

            if len(vx_sub)>0:

    #             print("vx_sub", vx_sub, r)

                dot_product = np.mean(vx_sub*vx_c + vy_sub*vy_c)

                if dot_product==0:
                    print("dot_product", dot_product)
                    pass
                else:
                    denominator = np.mean((vx_c**2+vy_c**2)**0.5)*np.mean((vx_sub**2+vy_sub**2)**0.5)
                    corr = dot_product/denominator
        #             print("center cell v: ", vx_c, vy_c)
        #             print("vs: ", vx_sub, vy_sub)
        #             print("dot: ", dot_product, "deno: ", denominator)
        #             print("corr: ", corr)
                    r_list.append(r)
                    corr_list.append(corr)
                    print(r, corr)
            else:
                pass

            r = r+dr

        r_list = np.asarray(r_list)
        corr_list = np.asarray(corr_list)

    #     print("r_list", r_list)
    #     print("corr_list", corr_list)


        # Fit an exponential decay function to find correlation length
        def exp_decay(r, L):
            return np.exp(-r / L)

        popt, _ = curve_fit(exp_decay, r_list, corr_list, p0=[10])

        correlation_length = popt[0]

        # Plot results

        plt.figure(figsize=(5, 3))
        plt.grid()
        plt.scatter(r_list, corr_list, label="Data", color='b', lw = 0, s = 22)
        plt.plot(r_list, exp_decay(r_list, *popt), label="Fit: $D_{corr}$="+f"{correlation_length:.2f}", color='r')
        plt.ylim((-1,1))
        plt.xlim((0,80))
        plt.xlabel("Distance $D_{toccell}$, pixel")
        plt.ylabel("Velocity Correlation")
        plt.legend(frameon = False, facecolor='white', framealpha=1, loc = (0.6,1))

        plt.savefig(figure_folder + name+ "_correlation.png", dpi = 200, bbox_inches = "tight")

    print("correlation_length: ", correlation_length)
    

    # ----------------------- calculate polarity properties -----------------------
    # -----------------------whole trajectory
    (past_px, past_py, past_fx, past_fy, past_angle) = read_time_series(90, 270, 5, 
                                                                        data_folder, rep_inx, iter_inx)
    #print("shape is: ", past_px.shape)
    (p_mean_180, cell_p_var_180) = polarity_mean(past_px, past_py)
    
    print("p_mean: %s, cell_p_var: %s"%(p_mean_180, cell_p_var_180))
    
    #-----------------------Final 30 frame
    (past_px, past_py, past_fx, past_fy, past_angle) = read_time_series(240, 270, 5, 
                                                                        data_folder, rep_inx, iter_inx)
    #print("shape is: ", past_px.shape)
    (p_mean_30, cell_p_var_30) = polarity_mean(past_px, past_py)
    
    print("p_mean: %s, cell_p_var: %s"%(p_mean_30, cell_p_var_30))

    


    
    
    if timepoint>180:
        
        hist_traj = cu.traj_file(data_folder, rep_inx, iter_inx, timepoint-180)
        hist_df = pd.read_csv(hist_traj)
        
        dxy = ((df.x- hist_df.x)**2+(df.y-hist_df.y)**2)**0.5
        
        dxy_mean = np.mean(dxy)
        dxy_std = np.std(dxy)
    else:
        dxy_mean = 0
        dxy_std = 0
    "Calc geo properties"

    # convex hull
    points = np.array([df.x, df.y]).T
    hull = ConvexHull(points)
    hull_canvas = np.zeros((256,256))
    

    # Delaunay
    tri = Delaunay(points)

    plt.figure(figsize = (9,9))
    ax = plt.subplot(111, aspect='equal')

    ax.scatter(df.x, df.y, s=20, c="b")
    ax.scatter(128,128, s=400, c="Red", marker="o")
    delaunay_plot_2d(tri, ax=ax)

    ax.set_xlim((0,256))
    ax.set_ylim((0,256))

    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-.', lw=2, alpha = 1)
    
    canvas = np.zeros((256,256))
    for simplex in tri.simplices:
        # distances for three edges
        xs = points[simplex, 0]
        ys = points[simplex, 1]
        edge1 = ((xs[0]-xs[1])**2+(ys[0]-ys[1])**2)**0.5
        edge2 = ((xs[0]-xs[2])**2+(ys[0]-ys[2])**2)**0.5
        edge3 = ((xs[1]-xs[2])**2+(ys[1]-ys[2])**2)**0.5
        #print(edge1, edge2, edge3)
        if edge1>12 or edge2>12 or edge3>12:
            continue

        rr, cc = polygon(points[simplex, 0], points[simplex, 1])
        canvas[rr, cc]=1

        # draw a triangle every time.
        # ax.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha = 0.5)
        # ax.plot(points[[simplex[0], simplex[2]], 0], points[[simplex[0], simplex[2]], 1], 'k-', alpha = 0.5)
        #break

    plt.imshow(canvas.T, origin='lower', cmap="tab20c")

    label_canvas = label(canvas)
    regions = regionprops(label_canvas)


    # find the index for largest area -- tumor body
    area_list = []
    peri_list = []
    i = 0
    while i<len(regions):
        area_list.append(regions[i].area)
        peri_list.append(regions[i].perimeter)
        i = i+1
    i_max = np.argmax(area_list)

    # 1.area - tumor total area, including debris
    area = np.sum(area_list)
    area_frac = area/256./256.
    
    # tumor area with total convex hall, including debris
    vetices = points[hull.vertices]
    area_convex = cu.PolyArea(points[hull.vertices,0],points[hull.vertices,1])
    area_convex_frac = area_convex/256./256.
    mode = cu.mode_judge_2(d, s, area_convex_frac, 32, 12)
    
    rr, cc = polygon(points[hull.vertices,0],points[hull.vertices,1])
    hull_canvas[rr, cc]=1
    label_hull_canvas = label(hull_canvas)
    hull_regions = regionprops(label_hull_canvas)
    #plt.figure()
    #plt.imshow(label_hull_canvas)
    
    # 2.perimeter
    # collecting perimeter pixel
    peri_x, peri_y = perimeter_coords(canvas)
    peri_hull_x, peri_hull_y = perimeter_coords(hull_canvas)
#     peri = np.sum(peri_list)
#     peri_convex = perimeter(hull_canvas)
    peri = float(len(peri_x))
    peri_convex = float(len(peri_hull_x))
    
    # 3. major axis
    major_axis = hull_regions[0].axis_major_length
    # 4. minor axis
    minor_axis = hull_regions[0].axis_minor_length
    minor_major_ratio = minor_axis/major_axis
    # 5. compactness 4pi*a/peri^2
    compact = (4*np.pi*area)/(peri**2)
    # 6. eccentricity
    eccentric= hull_regions[0].eccentricity
    # 7. Circularity
    circular = 4*np.pi*area/(peri_convex**2)
    # 8. convexity
    convexity = peri_convex/peri
    # 10. Solidity
    solidity = area/area_convex
#     # 11. normalized monents 
#     norm_moment = regions[i_max].moments_normalized[1,1]
    # 12. Radial distance entropy
    radial_dist_cx = ((peri_x-df.x.mean())**2+(peri_y-df.y.mean())**2)**0.5
    #radial_dist_cx_entropy = entropy(radial_dist_cx)
    radial_dist_cx_std = np.std(radial_dist_cx)
    radial_dist_cn = ((peri_x-128)**2+(peri_y-128)**2)**0.5
    radial_dist_cn_mean = np.mean(radial_dist_cn)
    radial_dist_cn_std = np.std(radial_dist_cn)
    # 13. sphericity
    sphericity = radial_dist_cx.min()/radial_dist_cx.max()
    
    # ------------- save data ------------------------------
    
    feature_dict = {
    "rep_inx": rep_inx,
    "iter_inx": iter_inx, 
    "timepoint": timepoint,

    "scan_f": scan_f,
    "scan_inx": scan_inx,
    "jasn_file": jasn_file,
    "scan_keys": scan_keys,
    "scan_param" : scan_param,
    "traj": traj,
    "mode_temp": mode,
    "region_n": len(regions),

    "rac_max": rac_max,
    "rac_mea": rac_mea,
    "s": s,
    "d": d,
    "d_std": d_std,
    "dxy_mean": dxy_mean,
    "dxy_std": dxy_std,
    "aa": aa,
    "ss": ss,
    "f_coef": f_coef,
    "force_max": force_max,
    "force_mea": force_mea,
    "fpp": fpp,
    "p_frac": p_frac,
    "ang_std": ang_std,
    "ang_mea": ang_mea,
    "polarity": polarity,

    "area": area,
    "area_convex": area_convex,
    "area_frac": area_frac,
    "area_convex_frac": area_convex_frac,
    "peri": peri,
    "peri_convex": peri_convex,
    "major_axis": major_axis,
    "minor_axis": minor_axis,
    "minor_major_ratio": minor_major_ratio,
    "compact": compact,
    "eccentric": eccentric,
    "circular": circular,
    "convexity": convexity,
#     "curl": curl,
    "solidity": solidity,
#     "norm_moment": norm_moment,
    "radial_dist_cx_std": radial_dist_cx_std,
    "radial_dist_cn_mean": radial_dist_cn_mean,
    "radial_dist_cn_std": radial_dist_cn_std,
    "sphericity": sphericity,
        
    "p_mean_180":p_mean_180,
    "cell_p_var_180":cell_p_var_180,

    "p_mean_30":p_mean_30,
    "cell_p_var_30":cell_p_var_30,
        
    "correlation_length":correlation_length,
        
    }
    #print(dxy_mean, dxy_std, radial_dist_cn_mean, radial_dist_cn_std)
    
    feature_df = pd.DataFrame(feature_dict, index=[0])
    #print(feature_df)
    #print(len(peri_x), len(peri_hull_x))
    
    name = "rep_"+"%s"%rep_inx+"_scan_"+"%s"%scan_inx+"_time_"+"%s"%timepoint
    feature_df.to_csv(feature_folder+name+".csv")
    plt.savefig(figure_folder + name+".png", dpi = 200, bbox_inches="tight")

    return feature_folder+name+".csv"

    
# data_folder, (str)
# feature_folder, (str)
# figure_folder, (str)
# rep_inx, (int)
# iter_inx, (int)
# timepoint, (int)

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    tumor_geo_calc(sys.argv[1], sys.argv[2], sys.argv[3], 
                   int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    """
    data_folder - string
    feature_folder - string
    figure_folder - string
    rep_inx - int
    iter_inx - int
    timepoint - [int,int] timepoint at the start and end of secretome treatment.
    """