import folium
from .database import query_neo4j


def visualize_spots(map_lat=49.4839211, map_lon=8.47808182):
    query = """
    MATCH (n:Spot)
    RETURN n
    """
    map_osm = folium.Map(location=[map_lat, map_lon])
    for res in query_neo4j(query):
        tup = (res['n']['latitude'], res['n']['longitude'])
        marker = folium.Marker(tup, popup=res['n']['spotID'])
        map_osm.add_children(marker)

    return map_osm


def visualize_cluster(df):
    from matplotlib.colors import cnames
    colors = [hexc for hexc in cnames.values()]
    map_lat, map_lon = df.iloc[0].start_lat, df.iloc[0].start_lon
    map_osm = folium.Map(location=[map_lat, map_lon])

    for i, row in df.iterrows():
        tup = (row['start_lat'], row['start_lon'])
        start_c = row['start_cluster']
        marker = folium.CircleMarker(tup, color=colors[start_c], fill_color=colors[start_c], radius=50, fill_opacity=1)
        map_osm.add_children(marker)

        tup = (row['end_lat'], row['end_lon'])
        end_c = row['end_cluster']
        marker = folium.CircleMarker(tup, color=colors[end_c], fill_color=colors[end_c], radius=50, fill_opacity=1)
        map_osm.add_children(marker)

    return map_osm


def visualize_rows(rows):
    map_lat, map_lon = rows.iloc[0].start_lat, rows.iloc[0].start_lon
    map_osm = folium.Map(location=[map_lat, map_lon])

    for i, row in rows.iterrows():
        tup = (row['start_lat'], row['start_lon'])
        marker = folium.Marker(tup, icon=folium.Icon(color='green'), popup='{} {}'.format(i, row['start_cluster']))
        map_osm.add_children(marker)
        tup = (row['end_lat'], row['end_lon'])
        marker = folium.Marker(tup, icon=folium.Icon(color='red'), popup='{} {}'.format(i, row['end_cluster']))
        map_osm.add_children(marker)

    return map_osm