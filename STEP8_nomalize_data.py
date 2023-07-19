import geopandas as gpd

if __name__== "__main__":

    gdf_pud=gpd.read_file("PUD.shp")
    print(gdf_pud.columns)
    independent_variables=["dist-train","trial_len", "hotels", "Com-nodes","protected",
                           "dem","HSOL_SUM_1","PRED_AVG_1","TA_AVG_1.5","PP_SUM_1.5","VV_AVG_2m",
                           "viewpoint","beach","restaurant","bike_len","nautic","lighthouse",
                           "town (m)","city (m)","arable","built","crops","forest","grassland","intertidal",
                           "marshes","rocks","scrub"]

    for variable in independent_variables:
        gdf_pud[variable]=(gdf_pud[variable]-gdf_pud[variable].mean())/gdf_pud[variable].mean()
    
    print(gdf_pud.describe())
    gdf_pud.to_file("PUD.shp",index=False)