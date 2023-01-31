from cc3d.core.PySteppables import *
import numpy as np
import pandas as pd
from scipy.spatial import distance

# invariant
relax_step = 120
switch_step = 210

# variable
a_init = 1.
s_max = 4.
lambda_fpp_init = 20
lambda_fpp_base = 1000
p_frac_scaler = 0.4 # scaling the neighbor polarity sharing.

# scan parameter
lambda_fpp_glob = 20
force_strenth_max = 150

rac_x = 30.
rac_thre = 40. # protrusion force threshold

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def init_rac(x = 30.):
    return np.random.uniform(0., x)


# define rac dynamics
model_string = '''
  // Reactions
  J1: -> rac; a+s;
  J2: rac -> ; d*rac;

  // Species initializations
  rac = 2;

  // Variable initialization
  a = %s;
  s = 0;
  
  d = 0.1;
''' %(a_init)

class migration_racdir_fppSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """

        for cell in self.cell_list:
            # posittion storage
            cell.dict["x_list"] = "0"
            cell.dict["y_list"] = "0"
            
            cell.dict["x_self_polarity"] = 0
            cell.dict["y_self_polarity"] = 0
            cell.dict["self_polarity"] = 0.
            

    def step(self,mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        
        # obtain secretor field
        secretome = self.get_field_secretor("secretome")
            
        # past xys
        # update x_list, y_list
        for cell in self.cell_list:
            # update cell.dict x,y
            cell.dict["x"] = cell.xCOM
            cell.dict["y"] = cell.yCOM
            #print(cell.dict["x_list"])
            cell.dict["x_list"] = cell.dict["x_list"]+" %s"%(cell.xCOM)
            cell.dict["y_list"] = cell.dict["y_list"]+" %s"%(cell.yCOM)

        # ---------------------------------------
        if mcs > relax_step:
            # update x_list, y_list length
            # update polarity vector
            for cell in self.cell_list:
                # calculate polarity and angle
                # previous position
                x = np.array(cell.dict["x_list"].split(" ")).astype(float)[-5]
                y = np.array(cell.dict["y_list"].split(" ")).astype(float)[-5]

                # calc polarity vector
                x_self_polarity = cell.dict["x"] - x
                y_self_polarity = cell.dict["y"] - y
                cell.dict["self_polarity"] = (x_self_polarity ** 2 + y_self_polarity ** 2) ** 0.5  # before norm
                norms = normalize([x_self_polarity, y_self_polarity])
                x_self_polarity = norms[0]
                y_self_polarity = norms[1]
                cell.dict["x_self_polarity"] = x_self_polarity
                cell.dict["y_self_polarity"] = y_self_polarity

        if switch_step > mcs > relax_step:
            # 150 steps, no secretome signals
            for cell in self.cell_list:
                cell.dict['s'] = 0.

        # Run once
        elif mcs == switch_step:
            # calc multiply factor
            # assign for secretome signal
            res_list = []
            for cell in self.cell_list:
                # cell, cap uptake, relative uptake
                res = - secretome.uptakeInsideCellTotalCount(cell, 1, .01/4.).tot_amount / cell.volume
                res_list.append(res)
                #print(dir(res))
                #print('secreted ', res, ' inside cell')

            #print(np.min(res_list), np.max(res_list))
            # s_max controls the max s cells can get.
            multiply = s_max/np.max(res_list)
            
            for cell in self.cell_list:
                cell.dict["multiply"]=multiply
        
        # Update every step
        elif mcs>switch_step:
            # update s signal for each cell
            res_list = []
            for cell in self.cell_list:
                # arguments are: cell, max uptake, relative uptake
                res = - secretome.uptakeInsideCellTotalCount(cell, 1, .01/4.).tot_amount / cell.volume
                cell.dict['s'] =res*cell.dict["multiply"]
                #cell.dict['s'] =0.
                res_list.append(res)
            #print(np.min(res_list)*cell.dict["multiply"], np.max(res_list)*cell.dict["multiply"])
            # -------------------------------------------------------
            
    def finish(self):
        """
        Finish Function is called after the last MCS
        """

    def on_stop(self):
        # this gets called each time user stops simulation
        return

class OdeSteppable(SteppableBasePy):
    
    # frequency is once every n step. 
    # The frequency can't be smaller than 1
    
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)


        self.track_cell_level_scalar_attribute(field_name='rac', attribute_name='rac')
        self.track_cell_level_scalar_attribute(field_name='s', attribute_name='s')
        self.track_cell_level_scalar_attribute(field_name='x_self_polarity', attribute_name='x_self_polarity')
        self.track_cell_level_scalar_attribute(field_name='y_self_polarity', attribute_name='y_self_polarity')
        self.track_cell_level_scalar_attribute(field_name='self_polarity', attribute_name='self_polarity')
        self.track_cell_level_scalar_attribute(field_name='f_x', attribute_name='f_x')
        self.track_cell_level_scalar_attribute(field_name='f_y', attribute_name='f_y')
        self.track_cell_level_scalar_attribute(field_name='f', attribute_name='f')
        self.track_cell_level_scalar_attribute(field_name='pol', attribute_name='pol')
        self.track_cell_level_scalar_attribute(field_name='f_coef', attribute_name='f_coef')
        self.track_cell_level_scalar_attribute(field_name='p_frac', attribute_name='p_frac')
        

    def start(self):
        # step size is not for frequency
        # step size is simulation times.
        # 24/30*3600 = 2880
        self.add_antimony_to_cell_types(model_string=model_string, model_name='RR', cell_types=[self.CELL], step_size = 2880)

        # ----------------------------------
        # initializing rac interaction
        # initializing cell x,y in dict
        for cell in self.cell_list:
            cell.dict["rac"] = 0.
            cell.dict["s"] = 0.

            cell.sbml.RR['rac'] = init_rac(rac_x)
            
            # motility using external potential.
            cell.lambdaVecX = 0.
            # force component pointing along Y axis
            cell.lambdaVecY = 0.

        self.timestep_sbml()

    def step(self, mcs):

        if mcs>relax_step:
            # ---------------------------------------
            for cell in self.cell_list:
                # update environment
                cell.sbml.RR['s'] = cell.dict['s']

            # ---------------------------------------
            # Run simulation once
            self.timestep_sbml()
            
            for cell in self.cell_list:
            # update simulation results
                cell.dict['rac'] = cell.sbml.RR['rac']
                cell.dict['a'] = cell.sbml.RR['a']


class FocalPointPlasticityParams(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):

        for cell in self.cell_list:

            lambda_fpp = 0.
            if switch_step>mcs>=relax_step:
                lambda_fpp = lambda_fpp_init
            elif mcs>=switch_step:
                lambda_fpp = lambda_fpp_glob

            cell.dict["fpp"] = lambda_fpp

            for fppd in self.get_focal_point_plasticity_data_list(cell):
                self.focalPointPlasticityPlugin.setFocalPointPlasticityParameters(cell, fppd.neighborAddress,
                                                                                  lambda_fpp, 8, 12)


class OdeUpdateParams(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        if mcs>relax_step:
            
            # Interaction with neighbors
            neighbor_x_list = []
            neighbor_y_list = []
            for cell in self.cell_list:
                neighbor_x_self = []
                neighbor_y_self = []
                for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                    if neighbor:
                        # print("neighbor.id", neighbor.id, " common_surface_area=", common_surface_area)
                        neighbor_x_self.append(neighbor.dict["x_self_polarity"])
                        neighbor_y_self.append(neighbor.dict["y_self_polarity"])
                    else:
                        # print("Medium common_surface_area=", common_surface_area)
                        pass
                if len(neighbor_x_self)==0:
                    mean_neighbor_x_self = 0.
                    mean_neighbor_y_self = 0.
                else:
                    mean_neighbor_x_self = np.mean(neighbor_x_self)
                    mean_neighbor_y_self = np.mean(neighbor_y_self)
                neighbor_x_list.append(mean_neighbor_x_self)
                neighbor_y_list.append(mean_neighbor_y_self)

            # cell-cell interaction
            lambda_fpp = cell.dict["fpp"]
            p_frac = lambda_fpp / lambda_fpp_base
            if p_frac >= 1.:
                p_frac = 1.
            p_frac = p_frac*p_frac_scaler

            i = 0
            for cell in self.cell_list:
                cell.dict["p_frac"] = p_frac
                
                x_self_polarity = p_frac * neighbor_x_list[i] + (1 - p_frac) * cell.dict["x_self_polarity"]
                y_self_polarity = p_frac * neighbor_y_list[i] + (1 - p_frac) * cell.dict["y_self_polarity"]
                norms = normalize([x_self_polarity, y_self_polarity])
                cell.dict["x_self_polarity"] = norms[0]
                cell.dict["y_self_polarity"] = norms[1]
                i = i+1
            
            
            for cell in self.cell_list:
                
                n = -force_strenth_max
                x_pol = cell.dict["x_self_polarity"]
                y_pol = cell.dict["y_self_polarity"]
                pol = (x_pol**2+y_pol**2)**0.5
                cell.dict["pol"] = pol
                
                cell.dict["f_coef"] = (cell.dict["rac"]**4)/(rac_thre**4+cell.dict["rac"]**4)
                cell.dict["f_x"] = n*cell.dict["f_coef"]*x_pol
                cell.dict["f_y"] = n*cell.dict["f_coef"]*y_pol
                cell.dict["f"] = force_strenth_max*cell.dict["f_coef"]
                # force component pointing along X axis
                cell.lambdaVecX = cell.dict["f_x"]
                # force component pointing along Y axis
                cell.lambdaVecY = cell.dict["f_y"]


            name = "%s_" % mcs + "a_%s_" % a_init + "s_%s" % s_max + "_record"
            name = name.replace(".", "_") + ".txt"
            file_obj, file_path = self.open_file_in_simulation_output_folder(name, mode='a')

            file_obj.write("cell_id,x,y,x_self_polarity,y_self_polarity,a,s,rac,f,f_x,f_y,fpp,f_coef,p_frac\n")
            for cell in self.cell_list_by_type(self.CELL):
                file_obj.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                    cell.id,
                    cell.xCOM,
                    cell.yCOM,
                    cell.dict["x_self_polarity"],
                    cell.dict["y_self_polarity"],

                    cell.dict["a"],
                    cell.dict["s"],
                    cell.dict["rac"],

                    cell.dict["f"],
                    cell.dict["f_x"],
                    cell.dict["f_y"],
                    cell.dict["fpp"],
                    cell.dict["f_coef"],
                    cell.dict["p_frac"]
                ))

            file_obj.close()
            print(file_path)