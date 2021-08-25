import spotipy
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import os
import json
import time
import functools
import powerlaw as pl
from spotipy.oauth2 import SpotifyClientCredentials
from sqlalchemy import create_engine
from sqlalchemy.dialects import registry
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas, pd_writer
from func_timeout import func_timeout
plt.rcParams['figure.figsize'] = [18, 12]
registry.register('snowflake', 'snowflake.sqlalchemy', 'dialect')

def bfs(num_artists):
    start = time.time()
    id_ = '6M2wZ9GZgrQXHCFfjv46we'
    with open('spotify_api_cred.json') as f:
        creds = json.load(f)
        auth_manager = SpotifyClientCredentials(client_id = creds['client_id'],
                                        client_secret= creds['client_secret'])
        sp = spotipy.Spotify(auth_manager=auth_manager)
    #print(sp.artist(id_))
    visited_artists = []
    added_artists = [(id_,sp.artist(id_)["name"],sp.artist(id_)["popularity"])]
    artist_dict = {}
    cnt = 1
    queue = [(id_,sp.artist(id_)["name"],sp.artist(id_)["popularity"],tuple(sp.artist(id_)["genres"]),sp.artist(id_)["followers"]["total"])]
    G = nx.Graph()
    
    try:
        os.remove(f"artist_conectivity_dict_{num_artists}artists.json")
        os.remove(f"artist_dict_{num_artists}artists.json")
    except:
        pass
    
    open(f"artist_conectivity_dict_{num_artists}artists.json", "a+")
    open(f"artist_dict_{num_artists}artists.json", "a+")
        
    required_fields = ["id","name","popularity","genres","followers"]
    while queue and len(visited_artists) < num_artists:
        current_artist = queue.pop(0)
        data = {current_artist : [(item["id"],item["name"],item["popularity"],tuple(item["genres"]),item["followers"]["total"]) for item in sp.artist_related_artists(current_artist[0])['artists']]}
        with open(f"artist_conectivity_dict_{num_artists}artists.json" , "r+") as f:
            try:
                temp = json.load(f)
            except:
                temp = {}
                
        with open(f"artist_conectivity_dict_{num_artists}artists.json" , "w+") as f:
            temp[str(current_artist)] = data[current_artist]
            json.dump(temp,f)
            
        with open(f"artist_dict_{num_artists}artists.json", "r+") as f:
            try:
                temp = json.load(f)
            except:
                temp = {}
                
        with open(f"artist_dict_{num_artists}artists.json", "w+") as f:
            temp[current_artist[0]] = current_artist[1:]
            for item in data[current_artist]:
                temp[item[0]] = item[1:] 
            json.dump(temp,f)
        
        cnt += len(data[current_artist])
        for x in data[current_artist]:
            if x not in visited_artists:
                visited_artists.append(x)
                queue.append(x)
                
    artist_index_dict = {}
    with open(f"artist_dict_{num_artists}artists.json", "r") as f:
        temp = json.load(f)
        for i,item in enumerate(temp.keys()):
            artist_index_dict[item] = i
            temp[item] = [i] + temp[item]
            
    with open(f"artist_dict_{num_artists}artists.json", "w") as f:
        json.dump(temp,f)

    with open(f"artist_conectivity_dict_{num_artists}artists.json", "r") as f:
        temp = json.load(f)
        for artist in temp.keys():
            try:
                artist_tuple = str_to_tuple(artist)
                for nei in temp[artist]:
                    G.add_edge(artist_index_dict[artist_tuple[0]],artist_index_dict[nei[0]])
            except ValueError:
                pass
            
    nx.write_adjlist(G,f"artists_{num_artists}artists.adjlist", comments='#', delimiter=' ', encoding='utf-8')

    print(f"Retrieving data from Spotify took {time.time()-start} seconds.")


def get_embeddings(n_artists,n_dims):
    try:
        os.remove(f"artists_{n_artists}artists_{n_dims}dims.embedding")
    except:
        pass
    start = time.time()
    os.system(f"deepwalk --input artists_{n_artists}artists.adjlist --output artists_{n_artists}artists_{n_dims}dims.embedding --representation-size {n_dims} --undirected False")
    print(f'Graph embedding took {time.time()-start} seconds.')

def draw_network(n_artists):
    adj_list = nx.read_adjlist(f"artists_{self.n_artists}artists.adjlist")
    G = nx.Graph(adj_list)
    nx.draw(G,with_labels=False,node_size=100,alpha=.5,width=0.05)
    
class Embedding_Analysis():

    def __init__(self,n_artists,n_dims):
        self.n_artists = n_artists
        self.n_dims = n_dims
        #self.n_clusters = n_clusters
        self.n_plots = 10
        self.artist_dict = {}
        self.dict_constructor()
        self.df_constructor()
        self.cluster_constructor()
        self.assign_color()

    def dict_constructor(self):
        with open(f"artist_dict_{self.n_artists}artists.json", "r") as f:
            temp = json.load(f)

        for key in temp.keys():
            self.artist_dict[temp[key][0]] = [key]+temp[key][1:]

    def df_constructor(self):
        self.df = pd.DataFrame(columns=['indx',"name","popularity","genres","n_followers"]+['dim%d'%i for i in range(1,self.n_dims+1)])
        with open(f"artists_{self.n_artists}artists_{self.n_dims}dims.embedding") as f:
            next(f)
            for line in f:
                temp = line.split(' ')
                name, popularity, genres, n_followers = self.artist_dict[int(temp[0])][1], self.artist_dict[int(temp[0])][2], self.artist_dict[int(temp[0])][3], self.artist_dict[int(temp[0])][4]
                temp_dict = {"indx":int(temp[0]),"name":name,"popularity":popularity,"genres":genres,"n_followers":n_followers}|{"dim%d"%i:float(temp[i]) for i in range(1,self.n_dims)}|{"dim%d"%self.n_dims:float(temp[self.n_dims].strip('\n'))}
                self.df = self.df.append(temp_dict,ignore_index=True)

    def cluster_constructor(self):
        total_var = {}
        for k in range(10,501,50):
            kmeans = KMeans(n_clusters=k,random_state=0).fit(self.df[['dim%d'%i for i in range(1,self.n_dims+1)]].values)
            total_var[k] = kmeans.inertia_
        
        plt.plot(total_var.keys(),total_var.values())
        plt.show()
        self.n_clusters = int(input("Type the optimum number of clusters: "))
        
        kmeans = KMeans(n_clusters=self.n_clusters,random_state=0).fit(self.df[['dim%d'%i for i in range(1,self.n_dims+1)]].values)
        self.df['cluster_assignment'] = kmeans.labels_

    def assign_color(self):
        color_codes = list(sns.color_palette(n_colors=self.n_plots))+[None for _ in range(self.n_clusters-self.n_plots)]
        self.df['node_color'] = self.df['cluster_assignment'].apply(lambda x: color_codes[x])

    def plot_clustered_network(self):
        adj_list = nx.read_adjlist(f"artists_{self.n_artists}artists.adjlist")                                                                                                           
        G = nx.Graph(adj_list)
        color_dict = self.df[['indx','node_color']].set_index('indx',drop=True).to_dict()['node_color']
        node_clr = [color_dict[int(x)] for x in G.nodes()]
        nx.draw(G,with_labels=False,node_color=node_clr,node_size=100,alpha=.5,width=0.05)
        
    def genre_dist(self):
        genre_dict, genre_trunc_dict, cluster_color = [], [], []
        is_wmg = {i:0 for i in range(self.n_plots)}
        tot_artists = {i:0 for i in range(self.n_plots)}
        tot_followers = {i:0 for i in range(self.n_plots)}
        for c in range(self.n_plots):
            temp = {}
            cluster_color.append(self.df.loc[self.df['cluster_assignment']==c].iloc[0]['node_color'])
            cur_cluster = self.df.loc[self.df["cluster_assignment"]==c]
            for i, row in cur_cluster.iterrows():
                tot_artists[c] += 1
                tot_followers[c] += row["n_followers"]
                for genre in row["genres"]:
                    if genre not in temp:
                        temp[genre] = 1
                    else:
                        temp[genre] += 1
                if row["is_wmg"]:
                    is_wmg[c] += 1
        
            temp = dict(sorted(temp.items(), key=lambda item: item[1], reverse=True))
            trunc_temp = dict(sorted(temp.items(), key=lambda item: item[1], reverse=True)[:5])
            genre_dict.append(temp)
            genre_trunc_dict.append(trunc_temp)
        
        width = [3,1]
        height = [1 for _ in range(self.n_plots)]
        kw_gs = dict(width_ratios=width,height_ratios=height)

        fig, ax = plt.subplots(nrows=self.n_plots,ncols=2,gridspec_kw=kw_gs)
        for i, row in enumerate(ax):
            row[0].bar(range(len(genre_dict[i])), list(genre_dict[i].values()), align='center', color=cluster_color[i])
            top_genres = list(genre_trunc_dict[i].keys())
            row[0].text(.95,.95,f'Top genres: {top_genres}',transform=row[0].transAxes,ha='right',va='top')
            row[0].text(.95,.75,f'WMG covers {100*is_wmg[i]/tot_artists[i]:{0}.{4}}%',transform=row[0].transAxes,ha='right',va='top')
            row[0].text(.95,.55,f'Average # followers {round(tot_followers[i]/tot_artists[i]):,}',transform=row[0].transAxes,ha='right',va='top')
            row[0].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
            values = [genre_dict[i][key] for key in genre_dict[i].keys()]                                                                                                                                
            results = pl.Fit(values,verbose=False)
            results.plot_pdf(ax=row[1],original_data=True,color=cluster_color[i],marker='o')
            if i == 0:
                row[0].set_title('Number of artists vs. genre in each cluster')
                row[1].set_title('PDF of number of artists in a genre')
        plt.savefig(f"genres_{self.n_artists}artists_{self.n_dims}n_dims.png")
        plt.show()

    def to_snowflake(self):
        temp_df = self.df
        temp_df["genres"] = temp_df["genres"].map(lambda x:str(x))
        temp_df["node_color"] = temp_df["node_color"].map(lambda x:str(x))
        temp_df.columns = map(str.upper, temp_df.columns)

        u = ''
        p = ''
        server = ''
        authenticator = ''

        engine = create_engine('', connect_args={'authenticator': authenticator})
        temp_df.to_sql('spotify_artist_embedding', engine, index=False, if_exists='replace', schema="", method=functools.partial(pd_writer, quote_identifiers=False))
        

    def from_snowflake(self):
        u = ''
        p = ''
        server = ''
        authenticator = ''
        engine = create_engine('', connect_args={'authenticator': authenticator})
        query_str = ""
        with engine.connect() as con:
            self.wmg_artists = func_timeout(100,pd.read_sql,args=(query_str,con))

    def update_with_wmg(self):
        try:
            self.df['is_wmg'] = self.df.name.isin(self.wmg_artists.name).astype(bool)
        except:
            pass
            
def str_to_tuple(key):
    return eval(key)

if __name__ == "__main__":
    n_artists = 1000
    n_dims = 5
    
    bfs(n_artists)
    
    get_embeddings(n_artists,n_dims)

    E = Embedding_Analysis(n_artists,n_dims)
    
    E.from_snowflake()
    E.update_with_wmg()
    E.genre_dist()
