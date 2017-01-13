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
