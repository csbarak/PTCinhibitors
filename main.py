import numpy as np
import pandas as pd

scaffold={}

scaffold[1] = np.asarray([-0.6918042917276589, 1.2000906905833681, -0.00018912104917642595])
scaffold[2] = np.asarray([0.6880925147495109, 1.2064862792986977, 0.005302683671743814])
scaffold[3] = np.asarray([1.388774355664162, -0.00015453223984703983, -0.000572666705581642])
scaffold[4] = np.asarray([0.6883606832473914, -1.207138920797627, -0.0012890949676999867])
scaffold[5] = np.asarray([-0.6922428025311852, -1.2003444952913116, -0.0019142654104198102])
scaffold[6] = np.asarray([-1.3811804594022203, 0.001060978446719496, -0.0013375355388659416])
scaffold[7] = np.asarray([2.8675006177467126, -2.220337117083852e-16, 2.199418402879527e-18])
scaffold[8] = np.asarray([5.007837319105866, 0.839583722578636, 0.001165159223780272])
scaffold[9] = np.asarray([5.512332681506209, -0.4229766819307225, 0.0009116208437578784])
scaffold[10] = np.asarray([4.002426484263403, -1.3815520647124881, 1.1926223897340549e-18])
scaffold[11] = np.asarray([3.7086677280187668, 1.0245803597748813, 0.0007203888773897177])


#compute avg distance between pairs of atoms in structure, and std
def clusterGeoFeatures(cluster):


    avg_dist_between_pairs = 0.0
    std_dist_between_pairs = 0.0

    for i in range(0, len(cluster)):
        for j in range(i + 1, len(cluster)):
            avg_dist_between_pairs += np.linalg.norm(cluster[i] - cluster[j])

    avg_dist_between_pairs /= (len(cluster) * (len(cluster) - 1) / 2)

    for i in range(0, len(cluster)):
        for j in range(i + 1, len(cluster)):
            std_dist_between_pairs += (avg_dist_between_pairs - np.linalg.norm(cluster[i] - cluster[j])) ** 2

    std_dist_between_pairs /= (len(cluster)  * (len(cluster) - 1) / 2)

    return avg_dist_between_pairs, std_dist_between_pairs


def getAtomName(s):
    for i in range(len(s)):
        if str.isdigit(s[i]):
            return s[0:i]

    return s


#distance from cluster is the distance to the closest point so far in the cluster, including theoretical scaffold
def distToCluster(cluster_tbl,cluster_id, pt):

    dist = np.linalg.norm(pt - scaffold[cluster_id])

    cluster = cluster_tbl.get(cluster_id,[])

    for i in range(0, len(cluster)):
        if (np.linalg.norm(pt - cluster[i]) < dist):
            dist = np.linalg.norm(pt - cluster[i])

    return dist

def findCluster(cluster_tbl,loc):

    min_dist = distToCluster(cluster_tbl,1,loc)
    target_cluster = 1

    for i in range(2, 12):
        dist = distToCluster(cluster_tbl,i,loc)
        if (dist < min_dist):
            min_dist = dist
            target_cluster = i

    return target_cluster

def getScaffoldAtom(cluster_pts,cluster_id):

    cur = 0

    for i in range(1, len(cluster_pts)):
        if(np.linalg.norm(cluster_pts[i] - scaffold[cluster_id]) < np.linalg.norm(cluster_pts[cur] - scaffold[cluster_id])):
            cur = i

    return cur


def processAtomLine(line, cluster_tbl):
    global scaff_check

    tokens = line.split()
    loc = np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])])
    cluster_id = findCluster(cluster_tbl,loc)
    if(cluster_id not in cluster_tbl.keys()):
        cluster_tbl[cluster_id]=[]
    cluster_tbl[cluster_id].append(loc)

def extractMoleculeFeatures(cluster_tbl):

    features_list = []

    for i in range(1, 12):

        assert(i in cluster_tbl.keys())  #make sure that every theoretical scaffold location was matched.

        scaff_atom = getScaffoldAtom(cluster_tbl[i], i)
        del cluster_tbl[i][scaff_atom]  # scaffold atom does not take part in the calculations

        # check for every scaffold location if it was modified or not
        if (len(cluster_tbl[i]) == 0):
            features_list.append(0)
        else:
            features_list.append(1)
        # if there is a structure - extract geometric features
        if (len(cluster_tbl[i]) >= 2):
            avg, std = clusterGeoFeatures(cluster_tbl[i])
            features_list.append(avg)
            features_list.append(std)
        else:
            features_list.append(0)
            features_list.append(0)

    return features_list


############# MAIN ################# U

# feature_file = open(r"/Users/sdannyvi/Dropbox/InPrgoress/Akabayov 2018/aligned.mol2","r")
# bond_value_file = open(r"/Users/sdannyvi/Dropbox/InPrgoress/Akabayov 2018/summary_2.0.sort",'r')

def getDataMatrix(x_fname,y_fname):

    minNumAt = 100000
    maxNumAt = 0
    totalNumAt = 0
    currentNumAt = 0

    feature_file = open(x_fname, "r")
    bond_value_file = open(y_fname, 'r')

    mol_features = {}
    bond_vals = {}

    in_process = False

    mol_num = 0

    for line in feature_file:

        if 'MOLECULE' in line:
            cluster_tbl = {}

        elif 'ZINC' in line:
            mol_num += 1
            mol_name = line.rstrip()
            # print('processing ' + mol_name + ' ' + str(mol_num) + '\n')

        elif 'ATOM' in line:
            in_process = True

        elif 'BOND' in line:

            in_process = False
            features_list = extractMoleculeFeatures(cluster_tbl)
            mol_features[mol_name.strip()] = features_list

            minNumAt = min(currentNumAt, minNumAt)
            maxNumAt = max(currentNumAt, maxNumAt)
            totalNumAt += currentNumAt
            currentNumAt = 0

        elif in_process:
            processAtomLine(line, cluster_tbl)
            currentNumAt += 1

        else:
            continue

    ### Extract bond values per molecule #####

    for line in bond_value_file:

        mol_name = line.split('_')[0].strip()
        val = line.split(',')[4].strip()
        bond_vals[mol_name]=val

    ### Construct data frame: features + bond values ####

    for mol_name in mol_features.keys():

        features_list = mol_features[mol_name]
        val = bond_vals.get(mol_name,'NaN')
        features_list = [mol_name] + features_list + [float(val)]
        mol_features[mol_name] = features_list

 #   matr = []
 #   for feature_line in mol_features.values():
 #       vals = feature_line.split()
 #       num_vals = [float(x) for x in vals[1::]]
 #       row = [vals[0]] + num_vals
 #       matr.append(row)

    frame = pd.DataFrame(mol_features.values())
    frame.columns = ['NAME', 'LOC1', 'LOC1DIST', 'LOC1STD', 'LOC2', 'LOC2DIST', 'LOC2STD', 'LOC3', 'LOC3DIST', 'LOC3STD',
                     'LOC4', 'LOC4DIST', 'LOC4STD', 'LOC5', 'LOC5DIST', 'LOC5STD', 'LOC6', 'LOC6DIS',
                     'LOC6STD', 'LOC7', 'LOC7DIST', 'LOC7STD', 'LOC8', 'LOC8DIST', 'LOC8STD', 'LOC9', 'LOC9DIST', 'LOC9STD' ,'LOC10', 'LOC10DIST',
                     'LOC10STD', 'LOC11', 'LOC11DIST', 'LOC11STD','BOND']
    return frame


    ############### Write CSV Output ####################

    # output_file = open(r'/Users/sdannyvi/Downloads/zinc.csv','w')
    # output_file.write('NAME LOC1 LOC1DIST LOC1STD LOC2 LOC2DIST LOC2STD LOC3 LOC3DIST LOC3STD LOC4 LOC4DIST LOC4STD LOC5 LOC5DIST LOC5STD LOC6 LOC6DIST LOC6STD LOC7 LOC7DIST LOC7STD LOC8 LOC8DIST LOC8STD LOC9 LOC9DIST LOC9STD LOC10 LOC10DIST LOC10STD LOC11 LOC11DIST LOC11STD Bond\n')


    #
    #    output_file.write(feature_line+'\n')



