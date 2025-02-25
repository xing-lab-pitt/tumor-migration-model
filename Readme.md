# Protrusion force cooperate with cell-cell adhesion-induced local cell polarity alignment to regulate switching between radial and directional collective tumor migration

This is code repository for the paper: *Protrusion force cooperate with cell-cell adhesion-induced local cell polarity alignment to regulate switching between radial and directional collective tumor migration.*

## Folders
**s1_model**: Folder containing CompuCell3D models. 

- <ins>12222021_force_fpp_single</ins>: CompuCell3D model for simulating microtumor dynamics.
    
- <ins>12222021_force_fpp_scan</ins>: CompuCell3D model parameter scans.

**s2_model_simulation**: Folder containing code for submitting simulation jobs to the cluster. 

**s3_model_analysis**: Including microtumor migration mode clustering model and code for geometric analysis.

- *01_geo_feature_calc.ipynb*: microtumor geometric feature calculations.
- *02_two_step_clustering.ipynb*: microtumor migratory mode clustering.

- <ins>clustering_models</ins>: folder containing saved clustering models.
- <ins>20250129-msmm-model-analysis</ins>: folder containing clustering results for simulation results.

**s4_figures**: Folder containing code for generating figures in the paper. Code for generating figure 1, 3, 4, 5 are archived in individual folders.

- <ins>Figure1</ins>: Folder containing code showing enriched GO terms
- <ins>Figure3</ins>: Folder containing code for Figure 3.

  - *20221201_Figure3_props_polarity.ipynb*: calculate example tumor properties and polarity at the final time point.
  - *20250203_Figure3_scan_props.ipynb*：Plot microtumor emergent properties for parameter scan.
  - *20221201_Figure3_h.ipynb*: Plot microtumor migratory mode distribution in emergent properties space.

  - <ins>Directional</ins>: folder containing simulation results for a directionally migratory tumor
  - <ins>Radial</ins>: folder containing simulation results for a radially migratory tumor
  - <ins>Results</ins>: folder containing generated figures.

- <ins>Figure4_n_5</ins>: Folder containing code generating Figure 4 and 5. 

  - *20220329_Figure4_a.ipynb*: Plot migratory mode percentage in selected parameter range
  - *20220602_Figure4_cd.ipynb*: Plot selected tumor migratory properties
  - *20220426_Figure5_e.ipynb*: Plot migratory mode percentage in selected parameter range
  - *20221105_Figure5_ij.ipynb*: Plot selected tumor migratory properties

  - <ins>myosin_selected</ins>: folder containing selected simulation results for migratory tumor with and without myosin inhibition
  - <ins>akt2_selected</ins>: folder containing selected simulation results for migratory tumor with and without AKT2 inhibition


**s5_supplements**: Folder containing model, data and code for supplemental simulations, where we scanned coupling strength beta as free parameter. beta values simulated are 0, 0.3,0.6,1

- *20250128_simulation_scan_betas.ipynb*: Code for submitting simulation jobs to cluster
- *20250131_analysis_1_geo_feature_calc_beta_scans.ipynb*: Code for tumor geometric feature analysis
- *20250203-analysis-2-two-step-clustering-beta-scans.ipynb*: Code for clustering tumor migration code using previous trained clustering model
- *20250203_analysis_3_plots_beta_scans.ipynb*: Code for plotting the emergent properties for microtumors

- 20250129-update-msmm-model: Folder containing CompuCell3D models for scanning beta values
- 20250129-update-msmm-model-analysis: Folder containing clustering results for simulations.

**src**: Folder containing code modules that are frequently used in this paper. 