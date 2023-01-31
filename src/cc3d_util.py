import pandas as pd
import numpy as np
from numpy import *
from scipy.stats import pearsonr
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import scipy
import matplotlib.pyplot as plt
from skimage.morphology import square, dilation
from skimage import data, util
from skimage.measure import label, perimeter

import seaborn as sns
import os
import re
import json

import pipe_util2

"""
List of functions:

mode_judge_2(d, s, thre, thre_2 = 10)
update_mode_detail(df, D=32, S=10)

potts_analysis_150_force_akt
Scaning force and akt level. The rac regulates also fpp.

potts_analysis_150_fpp_akt
Scaning fpp and akt level. The rac regulates also fpp.

potts_analysis_150_fpp_pfrac
potts_analysis_150_force_fpp

multi_land_draw # bivariate spline
multi_land_draw_griddata # griddata, don't have to be regular grid.

"""
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def traj_file(data_folder, rep_inx, iter_inx, timepoint, model_key="formal5"):
    """
    Obtain the trajectory file path for a simulation folder, rep num, iter and time point
    """
    repeat_folders = pipe_util2.folder_file_num(data_folder)
    
    rep = repeat_folders[rep_inx]
    rep = pipe_util2.folder_verify(rep)
    iter_f = rep+"scan_iteration_%s"%iter_inx
    iter_f = pipe_util2.folder_verify(iter_f)
    
    traj_paths = pipe_util2.folder_file_num(iter_f, model_key)[0]
    traj_paths = pipe_util2.folder_file_num(traj_paths, "record")
    
    traj = traj_paths[timepoint]
    return traj

def scan_iter_dirs(data_folder, rep_inx, iter_inx):
    
    repeat_folders = pipe_util2.folder_file_num(data_folder)
    
    rep = repeat_folders[rep_inx]
    rep = pipe_util2.folder_verify(rep)
    
    iter_folder = rep+"scan_iteration_%s"%iter_inx
    iter_folder = pipe_util2.folder_verify(iter_folder)
    
    return iter_folder


def mode_judge_2(d, s, ara_frac, thre, thre_2 = 10, ara_thresh = 0.167):
    
    """
    1 - stay
    2 - radial
    3 - dir
    
    d - tumor center displacement
    s - cell displacement from center
    """
    mode = 1
    if d<thre and s<thre_2:
        mode = 1
    elif d>=thre and s>=thre_2:
        mode = 3
    elif d>=thre and s<thre_2:
        mode = 2
    elif d<thre and s>=thre_2:
        mode = 1
    else: 
        print("Unpredicted mode: d %s s %s area %s"%(d, s, ara_frac))
        
    if ara_frac>ara_thresh:
        mode = 2
        
    return mode


def update_mode_detail(df, D=32, S=10, ara_thresh = 0.167):
    # update the mode list based on new threshold. 
    i = 0
    mode_list = []
    while i<df.shape[0]:
        d = df.d_list[i]
        s = df.s_list[i]
        ara_frac = df.area_list[i]
        mode = mode_judge_2(d, s, ara_frac, D, S, ara_thresh) # 32 -> cell displacement 9 -> tumor displacement
        mode_list.append(mode)
        i = i+1
    df["mode_list_2"] = mode_list
    return df

def counts_from_df_w_mode_col(df, mode_col, scan_param1, scan_param2):
    # obtain the modes in order. 
    # scan_param2 will be the row.
    counts_list = []
    param1_len = len(df[scan_param1].unique())
    param2_len = len(df[scan_param2].unique())
    
    for a in sorted(df[scan_param1].unique()):
        for f in sorted(df[scan_param2].unique()):
            local_df = df.loc[(df[scan_param1]==a)&(df[scan_param2]==f)].copy()
            counts = [0,0,0]

            i = 0
            while i<len(local_df[mode_col].value_counts()):
                inx = local_df[mode_col].value_counts().index[i]-1
                counts[inx] = local_df[mode_col].value_counts().values[i]
                i = i+1
            counts = np.array(counts)
            counts = counts/counts.sum()
            counts = np.round(counts, 2)
            counts_list.append(counts)
    return counts_list, param1_len, param2_len

def hist_draw(figfolder, counts_list, param1_len, param2_len, name):
    
    figfolder = pipe_util2.folder_verify(figfolder
                                        )
    i = 0
    height = int(8./param2_len*param1_len)
    if height <1:
        height = 1
    fig = plt.figure(figsize = (8, height))
    while i<len(counts_list):

        fig.add_subplot(param1_len,param2_len,i+1)
        plt.bar([1,2,3], counts_list[i], 
                width = 0.67, color = ["#E95C47","#66C2A6", "#0548B2"], alpha = 0.95)
        plt.ylim(0,1)
        plt.xlim(0.5,3.5)

        if i+1<=param2_len*(param1_len-1) and i%param2_len!=0:
            plt.xticks([])
            plt.yticks([])
        elif i+1<=param2_len*(param1_len-1) and i%param2_len==0:
            plt.xticks([])
            plt.yticks([0,0.5,1], fontsize = int(7/param2_len*15))
        elif i+1>param2_len*(param1_len-1) and i%param2_len!=0:
            plt.xticks([1,2,3], ["Non-migration", "Radial", "Directional"], 
                   rotation=90, fontsize = int(7/param2_len*15))
            plt.yticks([])
        else:
            plt.xticks([1,2,3], ["Non-migration", "Radial", "Directional"], 
                   rotation=90, fontsize = int(7/param2_len*15))
            plt.yticks([0,0.5,1], fontsize = int(7/param2_len*15))

        i = i+1
        
    plt.savefig(figfolder+"%s.pdf"%name, dpi = 200, bbox_inches="tight")


def migration_land_prepare(counts_list):
    
    # total migration
    i = 0
    mig_list = []
    while i<len(counts_list):
        mg = counts_list[i][1]+counts_list[i][2]
        mig_list.append(mg)
        i = i+1
    mig_list = np.array(mig_list)
    
    # different modes
    i = 0
    nmg_list = []
    rad_list = []
    dir_list = []
    while i<len(counts_list):
        nm = counts_list[i][0]
        rg = counts_list[i][1]
        dg = counts_list[i][2]

        nmg_list.append(nm)
        rad_list.append(rg)
        dir_list.append(dg)
        i = i+1

    nmg_list = np.array(nmg_list)
    rad_list = np.array(rad_list)
    dir_list = np.array(dir_list)
    
    return mig_list, nmg_list, rad_list, dir_list

def multi_land_draw(mg_list, x, y, cmap, fig_folder, name):
    
    fig_folder = pipe_util2.folder_verify(fig_folder)
    z_mg = mg_list.reshape((len(x), len(y)))
    
    xnew, ynew = np.mgrid[x[0]:x[-1]:80j, y[0]:y[-1]:80j]
    
    lut_mg = RectBivariateSpline(x, y, z_mg, s=0)
    znew_mg = lut_mg.ev(xnew.ravel(),
                        ynew.ravel()).reshape((80,80)).T
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca(projection='3d')
    ax.plot_surface(xnew, ynew, znew_mg, cmap=cmap, 
                    rstride=1, cstride=1, alpha=1, antialiased=True)
    ax.view_init(50, 120)
    
    ax.set_xlabel("Max force η", fontsize = 16)
    ax.set_ylabel("Adhesion λ", fontsize = 16)
    ax.set_zlabel("Migration mode fraction", fontsize = 16)
    ax.set(zlim=[0, 1])
    ax.set_box_aspect(aspect = (1,1,1))
    
    plt.savefig(fig_folder+name+".pdf", dpi = 200, bbox_inches = "tight")
    


def multi_land_draw_griddata(mg_list, x, y, cmap, fig_folder, name, method = "cubic"):
    
    fig_folder = pipe_util2.folder_verify(fig_folder)
    #z_mg = mg_list.reshape((len(x), len(y)))
    points = []
    for i in x:
        for j in y:
            points.append([i, j])
    
    xnew, ynew = np.mgrid[x[0]:x[-1]:80j, y[0]:y[-1]:80j]
    
    print(np.array(points).shape, mg_list.shape)
    
    grid_mg = griddata(points, mg_list, (xnew, ynew), method=method).T
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca(projection='3d')
    ax.plot_surface(xnew, ynew, grid_mg, cmap=cmap, 
                    rstride=1, cstride=1, alpha=1, antialiased=True)
    
    
    ax.set_xlabel("Max force η", fontsize = 16)
    ax.set_ylabel("Adhesion λ", fontsize = 16)
    ax.set_zlabel("Migration mode fraction", fontsize = 16)
    ax.set(zlim=[0, 1])
    ax.set_box_aspect(aspect = (1,1,1))
    ax.view_init(50, 140)
    
    plt.savefig(fig_folder+name+".pdf", dpi = 200, bbox_inches = "tight")
    plt.show()
    


def potts_analysis_150_force_akt(figure_folder, data_folder, 
                                 rep_num, timepoint, 
                                 param1_len, param2_len, hue):
    
    # This function generate a panel for a specific time.
    # figure_folder = Figures_+sim_name
    
    figure_folder = pipe_util2.create_folder(figure_folder)
    data_folder = pipe_util2.folder_verify(data_folder)
    iter_total = param1_len*param2_len
    
    inx_list = [] # the scan number
    d_list = [] # the single cell displacement
    d_std_list = []
    s_list = [] # the tumor center displacemnet
    area_list = []
    rac_list = [] # the average rac level for the tumor
    f_coef_list = [] # average f_coef
    
    a_record = [] # average a
    s_record = [] # average s
    fpp_record = [] # fpp strength
    force_record = [] # average force 
    p_frac_record = [] # average p_frac
    
    akt_scan_list = []
    force_scan_list = []
    
    angle_std = [] # angle (-pi, pi) std
    angle_mea = []
    mode_list = [] # tumor mode
    
    

    r = 0
    while r<rep_num:
        
        height = int(20./param2_len*param1_len)
        if height <1:
            height = 1
        fig = plt.figure(figsize = (20, height))
                         
        iter_inx = 0
        while iter_inx<iter_total:
            
            # obtain the real scan index
            scan_f = scan_iter_dirs(data_folder, r, iter_inx)
            scan_f_name = os.path.basename(scan_f)
            scan_inx = scan_f_name.split("_")[2]
            scan_inx = int(scan_inx)
            
            # obtain trajectory path
            traj = traj_file(data_folder, r, iter_inx, timepoint)
            
            print("Iteration is ", scan_f_name)
            print("file is:", traj)

            jasn_file = pipe_util2.folder_file_num(scan_f)[1]
            with open(jasn_file) as f:
                data = json.load(f)
            force_scan = float(data["parameters"]["force_scan"])
            force_scan = np.round(force_scan, 2)
            akt_scan = float(data["parameters"]["akt_scan"])
            akt_scan = np.round(akt_scan, 2) # the scan of maxium force. 
            
            # get position for the plots 
            df = pd.read_csv(traj)
            rac = np.round(df["rac"].mean(),2)

            ax = fig.add_subplot(param1_len, param2_len, scan_inx+1)

            # tumor center displacement from center
            s = ((df.x.mean()-128)**2+(df.y.mean()-128)**2)**0.5
            s = np.round(s, 2)

            # -----------------------------------------------------
            
            # average single cell displacment from center
            d = (((df.x-128)**2+(df.y-128)**2)**0.5).mean()
            d = np.round(d, 2)
            # average single cell displacement from center std
            d_std = (((df.x-128)**2+(df.y-128)**2)**0.5).std()
            d_std = np.round(d, 2)
            
            # -----------------------------------------------------
            # tumor area with convex hall
            points = np.array([df.x, df.y]).T
            hull = ConvexHull(points)
            vetices = points[hull.vertices]
            ara = PolyArea(points[hull.vertices,0],points[hull.vertices,1])
            ara_frac = ara/256/256.
            ara_frac = np.round(ara_frac, 2)
            #print("area is: ", ara, ara_frac)
                         
            sns.scatterplot(data=df, x="x", y="y", hue = hue, 
                            palette = "Blues",
                            s = 15./param2_len*10, alpha = 0.7, ax = ax)
            ax.set_aspect("equal")
            
            mode = mode_judge_2(d, s, ara_frac, 32, 9)
            aa = np.round(df.a.max(), 2)
            ss = np.round(df.s.max(), 2)
            f_coef = np.round(df.f_coef.max(),2)
            
            force = np.round(df.f.max(),2) # average force for the tumor
            fpp = np.round(df.fpp.max(), 2) # average fpp 
            p_frac = np.round(df.p_frac.max(),2)
            
            px = df.x_self_polarity
            py = df.y_self_polarity
            ang = np.arctan2(py, px)
            ang_std = np.round(np.std(ang),2)
            ang_mea = np.round(np.mean(ang),2)
            
            inx_list.append(scan_inx)
            d_list.append(d)
            d_std_list.append(d_std)
            s_list.append(s)
            area_list.append(ara_frac)
            rac_list.append(rac)
            f_coef_list.append(f_coef)
            
            a_record.append(aa)
            s_record.append(ss)
            fpp_record.append(fpp)
            force_record.append(force)
            p_frac_record.append(p_frac)
            
            akt_scan_list.append(akt_scan)
            force_scan_list.append(force_scan)
            
            angle_std.append(ang_std) # angle (-pi, pi) std
            angle_mea.append(ang_mea)
            mode_list.append(mode)

            ax.plot([128],[128],'o', ms=30./param2_len*10, mec='Gray', mfc='none')
            
            ax.set_title("p_frac=%s,ang_std=%s,area=%s\nfpp=%s,f=%s\nakt_scan=%s,f_scan=%s\nrac=%s,D=%s,S=%s"%(
                         p_frac,ang_std, ara_frac, fpp, force, akt_scan, force_scan, rac, d, s), fontsize = 5/param2_len*10)
            ax.set_ylim(0,256)
            ax.set_xlim(0,256) 
            ax.set_ylabel(None)
            ax.set_xlabel(None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.get_legend().remove()
            
            iter_inx = iter_inx+1
            
        plt.savefig(figure_folder + "Supplementary_fpp_force_scan_%s.pdf"%r, dpi = 200, bbox_inches="tight")
        #plt.show()
        plt.close()
        r = r+1
    
#     print(len(angle_std), len(akt_record),
#           len(inx_list), len(d_list), len(s_list),
#           len(a_record), len(s_record), len(rac_list),
#           len(fpp_record), len(force_record), len(mode_list)
#          )
    prop_df = pd.DataFrame({
        
        "inx_list": inx_list,
        "d_list": d_list,
        "d_std_list": d_std_list,
        "s_list": s_list,
        "area_list": area_list,
        "rac_list": rac_list,
        "f_coef_list": f_coef_list,
        
        "a_record": a_record,
        "s_record": s_record,
        "fpp_record": fpp_record,
        "force_record": force_record,
        "p_frac_record": p_frac_record,
        
        "akt_scan_list": akt_scan_list,
        "force_scan_list": force_scan_list,
        
        "mode_list": mode_list,
        "angle_std" : angle_std, # angle (-pi, pi) std
        "angle_mea" : angle_mea

    })

    prop_df.to_csv(figure_folder + "properties.csv")
    
    return prop_df



def potts_analysis_150_fpp_pfrac(figure_folder, data_folder, 
                                 rep_num, timepoint, 
                                 param1_len, param2_len, hue):
    
    # This function generate a panel for a specific time.
    # figure_folder = Figures_+sim_name
    
    figure_folder = pipe_util2.create_folder(figure_folder)
    data_folder = pipe_util2.folder_verify(data_folder)
    iter_total = param1_len*param2_len
    
    inx_list = [] # the scan number
    d_list = [] # the single cell displacement
    s_list = [] # the tumor center displacemnet
    rac_list = [] # the average rac level for the tumor
    f_coef_list = [] # average f_coef
    
    a_record = [] # average a
    s_record = [] # average s
    fpp_record = [] # fpp strength
    force_record = [] # average force 
    p_frac_record = [] # average p_frac
    
    fpp_scan_list = []
    p_frac_scan_list = []
    
    angle_std = [] # angle (-pi, pi) std
    angle_mea = []
    mode_list = [] # tumor mode
    
    

    r = 0
    while r<rep_num:
        
        fig = plt.figure(figsize = (20, int(20./param2_len*param1_len)))
                         
        iter_inx = 0
        while iter_inx<iter_total:
            
            # obtain the real scan index
            scan_f = scan_iter_dirs(data_folder, r, iter_inx)
            scan_f_name = os.path.basename(scan_f)
            scan_inx = scan_f_name.split("_")[2]
            scan_inx = int(scan_inx)
            
            # obtain trajectory path
            traj = traj_file(data_folder, r, iter_inx, timepoint)
            
            print("Iteration is ", scan_f_name)
            print("file is:", traj)

            jasn_file = pipe_util2.folder_file_num(scan_f)[1]
            with open(jasn_file) as f:
                data = json.load(f)
            fpp_scan = float(data["parameters"]["fpp_scan"])
            fpp_scan = np.round(fpp_scan, 2)
            p_frac_scan = float(data["parameters"]["p_frac_scan"])
            p_frac_scan = np.round(p_frac_scan, 2) # the scan of maxium force. 
            
            # get position for the plots 
            df = pd.read_csv(traj)
            rac = np.round(df["rac"].mean(),2)

            ax = fig.add_subplot(param1_len, param2_len, scan_inx+1)

            # tumor center displacement from center
            s = ((df.x.mean()-128)**2+(df.y.mean()-128)**2)**0.5
            s = np.round(s, 2)

            # -----------------------------------------------------
            
            # average single cell displacment from center
            d = (((df.x-128)**2+(df.y-128)**2)**0.5).mean()
            d = np.round(d, 2)
                         
            sns.scatterplot(data=df, x="x", y="y", hue = hue, 
                            palette = "Blues",
                            s = 15./param2_len*10, alpha = 0.7, ax = ax)
            ax.set_aspect("equal")
            
            mode = mode_judge_2(d, s, 32, 9)
            aa = np.round(df.a.max(), 2)
            ss = np.round(df.s.max(), 2)
            f_coef = np.round(df.f_coef.max(),2)
            
            force = np.round(df.f.max(),2) # average force for the tumor
            fpp = np.round(df.fpp.max(), 2) # average fpp 
            p_frac = np.round(df.p_frac.max(),2)
            
            px = df.x_self_polarity
            py = df.y_self_polarity
            ang = np.arctan2(py, px)
            ang_std = np.round(np.std(ang),2)
            ang_mea = np.round(np.mean(ang),2)
            
            inx_list.append(scan_inx)
            d_list.append(d)
            s_list.append(s)
            rac_list.append(rac)
            f_coef_list.append(f_coef)
            
            a_record.append(aa)
            s_record.append(ss)
            fpp_record.append(fpp)
            force_record.append(force)
            p_frac_record.append(p_frac)
            
            fpp_scan_list.append(fpp_scan)
            p_frac_scan_list.append(p_frac_scan)
            
            angle_std.append(ang_std) # angle (-pi, pi) std
            angle_mea.append(ang_mea)
            mode_list.append(mode)

            ax.plot([128],[128],'o', ms=30./param2_len*10, mec='Gray', mfc='none')
            
            ax.set_title("p_frac=%s,ang_std=%s\nfpp=%s,f=%s\nfpp_scan=%s,p_frac_scan=%s\nrac=%s,D=%s,S=%s"%(
                         p_frac,ang_std, fpp, force, fpp_scan, p_frac_scan, rac, d, s), fontsize = 5/param2_len*10)
            ax.set_ylim(0,256)
            ax.set_xlim(0,256) 
            ax.set_ylabel(None)
            ax.set_xlabel(None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.get_legend().remove()
            
            iter_inx = iter_inx+1
            
        plt.savefig(figure_folder + "Supplementary_fpp_force_scan_%s.pdf"%r, dpi = 200, bbox_inches="tight")
        #plt.show()
        plt.close()
        r = r+1
    
#     print(len(angle_std), len(akt_record),
#           len(inx_list), len(d_list), len(s_list),
#           len(a_record), len(s_record), len(rac_list),
#           len(fpp_record), len(force_record), len(mode_list)
#          )
    prop_df = pd.DataFrame({
        
        "inx_list": inx_list,
        "d_list": d_list,
        "s_list": s_list,
        "rac_list": rac_list,
        "f_coef_list": f_coef_list,
        
        "a_record": a_record,
        "s_record": s_record,
        "fpp_record": fpp_record,
        "force_record": force_record,
        "p_frac_record": p_frac_record,
        
        "fpp_scan_list": fpp_scan_list,
        "p_frac_scan_list": p_frac_scan_list,
        
        "mode_list": mode_list,
        "angle_std" : angle_std, # angle (-pi, pi) std
        "angle_mea" : angle_mea

    })

    prop_df.to_csv(figure_folder + "properties.csv")
    
    return prop_df




def potts_analysis_150_force_fpp(figure_folder, data_folder, 
                                 rep_num, timepoint, 
                                 param1_len, param2_len, hue):
    
    # This function generate a panel for a specific time.
    # figure_folder = Figures_+sim_name
    
    figure_folder = pipe_util2.create_folder(figure_folder)
    data_folder = pipe_util2.folder_verify(data_folder)
    iter_total = param1_len*param2_len
    
    inx_list = [] # the scan number
    d_list = [] # the single cell displacement
    d_std_list = []
    s_list = [] # the tumor center displacemnet
    area_list = []
    rac_list = [] # the average rac level for the tumor
    f_coef_list = [] # average f_coef
    
    a_record = [] # average a
    s_record = [] # average s
    fpp_record = [] # fpp strength
    force_record = [] # average force 
    p_frac_record = [] # average p_frac
    
    fpp_scan_list = []
    force_scan_list = []
    
    angle_std = [] # angle (-pi, pi) std
    angle_mea = []
    polarity_list = [] # polarity
    mode_list = [] # tumor mode
    
    

    r = 0
    while r<rep_num:
        
        height = int(20./param2_len*param1_len)
        if height <4:
            height = 4
        fig = plt.figure(figsize = (20, height))
                         
        iter_inx = 0
        while iter_inx<iter_total:
            
            # obtain the real scan index
            scan_f = scan_iter_dirs(data_folder, r, iter_inx)
            scan_f_name = os.path.basename(scan_f)
            scan_inx = scan_f_name.split("_")[2]
            scan_inx = int(scan_inx)
            
            # obtain trajectory path
            traj = traj_file(data_folder, r, iter_inx, timepoint)
            
            print("Iteration is ", scan_f_name)
            print("file is:", traj)

            jasn_file = pipe_util2.folder_file_num(scan_f)[1]
            with open(jasn_file) as f:
                data = json.load(f)
            fpp_scan = float(data["parameters"]["fpp_scan"])
            fpp_scan = np.round(fpp_scan, 2)
            force_scan = float(data["parameters"]["force_scan"])
            force_scan = np.round(force_scan, 2) # the scan of maxium force. 
            
            # get position for the plots 
            df = pd.read_csv(traj)
            rac = np.round(df["rac"].mean(),2)

            ax = fig.add_subplot(param1_len, param2_len, scan_inx+1)

            # tumor center displacement from center
            s = ((df.x.mean()-128)**2+(df.y.mean()-128)**2)**0.5
            s = np.round(s, 2)

            # -----------------------------------------------------
            
            # average single cell displacment from center
            d = (((df.x-128)**2+(df.y-128)**2)**0.5).mean()
            d = np.round(d, 2)
            # average single cell displacement from center std
            d_std = (((df.x-128)**2+(df.y-128)**2)**0.5).std()
            d_std = np.round(d, 2)
            
            # -----------------------------------------------------
            # tumor area with convex hall
            points = np.array([df.x, df.y]).T
            hull = ConvexHull(points)
            vetices = points[hull.vertices]
            ara = PolyArea(points[hull.vertices,0],points[hull.vertices,1])
            ara_frac = ara/256/256.
            ara_frac = np.round(ara_frac, 2)
            #print("area is: ", ara, ara_frac)
                         
            sns.scatterplot(data=df, x="x", y="y", hue = hue, 
                            palette = "Blues",
                            s = 15./param2_len*10, alpha = 0.7, ax = ax)
            ax.set_aspect("equal")
            
            mode = mode_judge_2(d, s, ara_frac, 32, 9)
            aa = np.round(df.a.max(), 2)
            ss = np.round(df.s.max(), 2)
            f_coef = np.round(df.f_coef.max(),2)
            
            force = np.round(df.f.max(),2) # average force for the tumor
            fpp = np.round(df.fpp.max(), 2) # average fpp 
            p_frac = np.round(df.p_frac.max(),2)
            
            px = df.x_self_polarity
            py = df.y_self_polarity
            ang = np.arctan2(py, px)
            ang_std = np.round(np.std(ang),2)
            ang_mea = np.round(np.mean(ang),2)
            
            # polarity
            top = (px.sum()**2+py.sum()**2)**0.5
            bot = ((px**2+py**2)**0.5).sum()
            #print(top)
            if bot==0:
                polarity = 0.
            else:
                polarity = top/bot
            
            inx_list.append(scan_inx)
            d_list.append(d)
            d_std_list.append(d_std)
            s_list.append(s)
            area_list.append(ara_frac)
            rac_list.append(rac)
            f_coef_list.append(f_coef)
            
            a_record.append(aa)
            s_record.append(ss)
            fpp_record.append(fpp)
            force_record.append(force)
            p_frac_record.append(p_frac)
            
            fpp_scan_list.append(fpp_scan)
            force_scan_list.append(force_scan)
            
            angle_std.append(ang_std) # angle (-pi, pi) std
            angle_mea.append(ang_mea)
            polarity_list.append(polarity)
            mode_list.append(mode)

            ax.plot([128],[128],'o', ms=30./param2_len*10, mec='Gray', mfc='none')
            
            ax.set_title("p_frac=%s,ang_std=%s,area=%s\nfpp=%s,f=%s\nfpp_scan=%s,f_scan=%s\nrac=%s,D=%s,S=%s"%(
                         p_frac,ang_std, ara_frac, fpp, force, fpp_scan, force_scan, rac, d, s), fontsize = 5/param2_len*10)
            ax.set_ylim(0,256)
            ax.set_xlim(0,256) 
            ax.set_ylabel(None)
            ax.set_xlabel(None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.get_legend().remove()
            
            iter_inx = iter_inx+1
            
        plt.savefig(figure_folder + "Supplementary_fpp_force_scan_%s.pdf"%r, dpi = 200, bbox_inches="tight")
        #plt.show()
        plt.close()
        r = r+1
    
#     print(len(angle_std), len(akt_record),
#           len(inx_list), len(d_list), len(s_list),
#           len(a_record), len(s_record), len(rac_list),
#           len(fpp_record), len(force_record), len(mode_list)
#          )
    prop_df = pd.DataFrame({
        
        "inx_list": inx_list,
        "d_list": d_list,
        "d_std_list": d_std_list,
        "s_list": s_list,
        "area_list": area_list,
        "rac_list": rac_list,
        "f_coef_list": f_coef_list,
        
        "a_record": a_record,
        "s_record": s_record,
        "fpp_record": fpp_record,
        "force_record": force_record,
        "p_frac_record": p_frac_record,
        
        "fpp_scan_list": fpp_scan_list,
        "force_scan_list": force_scan_list,
        
        "mode_list": mode_list,
        "angle_std" : angle_std, # angle (-pi, pi) std
        "angle_mea" : angle_mea,
        "polarity_list": polarity_list

    })

    prop_df.to_csv(figure_folder + "properties.csv")
    
    return prop_df