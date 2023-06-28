import uuid
import duckdb
import dgl
import torch
import numpy as np
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import SparseTensor
from collections.abc import Iterable
"""
mem,
kuzu,
parallelized,

"""
###TODO:
#### 1. IntermediateGraph to DGL and PyG
#### 2. Wrapper to save DGL and PyG graph to DuckPGQ
class CSR:
    def __init__(self, v, e, w = None):
        self.v = v
        self.e = e
        self.w = w
        self.weighted = False
        if w:
            self.weighted = True

class GraphElementTable:
    def __init__(self, table_name, properties, label):
        self.table_name = table_name
        self.properties = properties
        self.label = label

class VertexTable:
    def __init__(self, table_name, properties, label):
        self.table_name = table_name
        self.properties = properties
        self.label = label

    def _to_query_expr(self):
        properties = ", ".join(self.properties)
        return "{} PROPERTIES ( {} ) LABEL {}".format(self.table_name, properties, self.label)

    def _validate(self):
        pass

class EdgeTable:
    def __init__(self, table_name, properties, label, src_key, src_ref_table, src_ref_col,  dst_key, dst_ref_table, dst_ref_col):
        self.table_name = table_name
        self.properties = properties
        self.label = label
        if type(src_key) == str:
            self.src_key = [src_key]
        else:
            self.src_key = src_key
        self.src_ref_table = src_ref_table
        if isinstance(src_ref_col, str):
            self.src_ref_col = [src_ref_col]
        else:
            self.src_ref_col = src_ref_col
        if isinstance(dst_key, str):
            self.dst_key = [dst_key]
        else:
            self.dst_key = dst_key
        self.dst_ref_table = dst_ref_table
        if isinstance(dst_ref_col, str):
            self.dst_ref_col = [dst_ref_col]
        else:
            self.dst_ref_col = dst_ref_col
    def _to_query_expr(self):
        properties = ", ".join(self.properties)
        assert len(self.src_key) == len(self.src_ref_col)
        assert len(self.dst_key) == len(self.dst_ref_col)
        src_key = ", ".join(self.src_key)
        src_ref_col = ", ".join(self.src_ref_col)
        dst_key = ", ".join(self.dst_key)
        dst_ref_col = ", ".join(self.dst_ref_col)
        return "{} SOURCE KEY ( {} ) REFERENCES {} ( {} ) DESTINATION KEY ( {} ) REFERENCES {} ( {} ) PROPERTIES ( {} ) LABEL {}".format(self.table_name,
                                                   src_key, self.src_ref_table, src_ref_col,
                                                   dst_key, self.dst_ref_table, dst_ref_col,
                                                   properties, self.label)

    def _validate(self):
        pass

class IntermediateGraph:
    """ The intermediate graph representation of a PG.
        TODO: Maybe it's helpful to seprate hetore and homo graphs?
    """

    def __init__(self, edges, edge_features, node_features):
        #if(type(edges) != CSR):
        #    raise Exception("Currently we don't support creating edges from {}".format(type(edges)))
        self.edges = edges

        self.edge_feature_names = {}
        self.edge_features = {}
        for tabk, tabv in edge_features.items():
            self.edge_features[tabk] = []
            self.edge_feature_names[tabk] = []
            for k, v in tabv.items():
                self.edge_feature_names[tabk].append(k)
                self.edge_features[tabk].append(v.data)

        self.node_feature_names = {}
        self.node_features = {}
        for tabk, tabv in node_features.items():
            self.node_features[tabk] = []
            self.node_feature_names[tabk] = []
            for k, v in tabv.items():
                self.node_feature_names[tabk].append(k)
                self.node_features[tabk].append(v.data)
        
        ## Heterogeneous Graph?
        if len(self.node_feature_names) == 1 and len(self.edge_feature_names) == 1:
            self.is_heterogeneous = False
        else:
            self.is_heterogeneous = True

    def as_pyg(self, drop_non_numerical = True):
        if self.is_heterogeneous:
            raise Exception("Exporting heterograph as PyTorch Geometric is not supported.")
        else:
            # transform CSR to SparseTensor
            csrv = torch.from_numpy(self.edges.v)
            csre = torch.from_numpy(self.edges.e)
            if self.edges.weighted:
                csrw = torch.from_numpy(self.edges.w)
                adj_t = SparseTensor(rowptr = csrv, col = csre, value = csrw)
            else:
                adj_t = SparseTensor(rowptr = csrv, col = csre)
            # prepare node features
            for k, v in self.node_feature_names.items():
                node_table = k
            names = self.node_feature_names[node_table]
            values = self.node_features[node_table]
            assert(len(names) == len(values))
            newnames = []
            newvalues = []
            for i in range(len(names)):
                if not np.issubdtype(values[i].dtype, np.number):
                    if drop_non_numerical:
                        continue
                    raise Exception("Failed to convert column {} of type {} to PyTorch.Tensor".format(node_table + '.' + names[i], values[i].dtype))
                else:
                    newnames.append(names[i])
                    newvalues.append(values[i])
            datax = torch.from_numpy(np.array(newvalues))
            # now edge features
            for k, v in self.edge_feature_names.items():
                edge_table = k
            names = self.edge_feature_names[edge_table]
            values = self.edge_features[edge_table]
            assert(len(names) == len(values))
            newnames = []
            newvalues = []
            for i in range(len(names)):
                if not np.issubdtype(values[i].dtype, np.number):
                    if drop_non_numerical:
                        continue
                    raise Exception("Failed to convert column {} of type {} to PyTorch.Tensor".format(node_table + '.' + names[i], values[i].dtype))
                else:
                    newnames.append(names[i])
                    newvalues.append(values[i])
            edge_attr = torch.from_numpy(np.array(newvalues))
            data = Data(x = datax, edge_attr = edge_attr)
            data.adj_t = adj_t
            # do we need to return feature names?
            return (data, newnames)

    def as_dgl(self, edge_id_column = None, drop_non_numerical = True):
        if self.is_heterogeneous:
            assert(type(self.edges) == list)
            if edge_id_column:
                assert(type(edge_id_column) == dict)
            graph_data = {}
            for elem in self.edges:
                assert(len(elem) == 3)
                edge_table, csr, key = elem
                if edge_id_column and edge_table in edge_id_column.keys() and edge_id_column[edge_table]:
                    id_col_index = self.edge_feature_names[edge_table].index[edge_id_column[edge_table]]
                    graph_data[key] = ( 'csr', ( csr.v, csr.e, self.edge_features[table_name][id_col_index] ) )
                else:
                    graph_data[key] = ( 'csr', ( csr.v, csr.e, [] ) )
            g = dgl.heterograph(graph_data)
            # now we prepare the node features
            for table_name, values in self.node_features:
                feature_names = self.node_feature_names[table_name]
                for i, feature in enumerate(feature_names):
                    try:
                        tensor = torch.from_numpy(values[i])
                    except TypeError:
                        if drop_non_numerical:
                            continue
                        raise Exception("Failed to convert column {} of type {} to PyTorch.Tensor".format(node_table + '.' + names[i], values[i].dtype))
                    g.nodes[table_name].data[feature] = tensor
            # now edge features
            for table_name, values in self.edge_features:
                feature_names = self.edge_feature_names[table_name]
                for i, feature in enumerate(feature_names):
                    if edge_id_column and table_name in edge_id_column.keys() and edge_id_column[table_name] == feature:
                        continue
                    try:
                        tensor = torch.from_numpy(values[i])
                    except TypeError:
                        if drop_non_numerical:
                            continue
                        raise Exception("Failed to convert column {} of type {} to PyTorch.Tensor".format(node_table + '.' + names[i], values[i].dtype))
                    g.edges[table_name].data[feature] = tensor
            for table_name, csr, key in self.edges:
                random_string = uuid.uuid1().hex
                if csr.w:
                    g.edges[table_name].data['weight_' + random_string] = torch.from_numpy(csr.w)
            return g
        else:
            assert(type(self.edges) == CSR)
            for k, v in self.edge_feature_names.items():
                edge_table = k
            if edge_id_column:
                id_col_index = self.edge_feature_names[edge_table].index(edge_id_column)
                g = dgl.graph( ( 'csr', ( self.edges.v[:-1], self.edges.e, self.edge_features[table_name][id_col_index] ) ) )
            else:
                g = dgl.graph( ( 'csr', ( self.edges.v[:-1], self.edges.e, [] ) ) )
            # now we prepare the nodes features
            for k, v in self.node_feature_names.items():
                node_table = k
            names = self.node_feature_names[node_table]
            values = self.node_features[node_table]
            assert(len(names) == len(values))
            for i in range(len(names)):
                try:
                    tensor = torch.from_numpy(values[i])
                except TypeError:
                    if drop_non_numerical:
                        continue
                    raise Exception("Failed to convert column {} of type {} to PyTorch.Tensor".format(node_table + '.' + names[i], values[i].dtype))
                g.ndata[names[i]] = tensor
            # now edge features
            for k, v in self.edge_feature_names.items():
                edge_table = k
            names = self.edge_feature_names[edge_table]
            values = self.edge_features[edge_table]
            assert(len(names) == len(values))
            for i in range(len(names)):
                if names[i] == edge_id_column:
                    continue
                try:
                    tensor = torch.from_numpy(values[i])
                except TypeError:
                    if drop_non_numerical:
                        continue
                    raise Exception("Failed to convert column {} of type {} to PyTorch.Tensor".format(node_table + '.' + names[i], values[i].dtype))
                g.edata[names[i]] = tensor
            # we add weights to edata
            if self.edges.w:
                # add a random string to avoid duplicate names
                g.edata['weight_' + uuid.uuid1().hex] = torch.from_numpy(self.edges.w)
            return g


def _gen_query(table_name, col_names):
    query = "SELECT {} FROM {};"
    cols = ", ".join(col_names)
    return query.format(cols, table_name)

def connect(path = ":memory:", read_only = False, config = {}):
    return Connection(path, read_only, config)

class Connection:
    """ A wrapper for DuckDB connections
        TODO: now we have this intermediate representation, maybe we hide some low-level methods?
    """

    def __init__(self, path = ":memory:", read_only = False, config = {}):
        self.con = duckdb.connect(path, read_only, config)

    def sql(self, query, alias = "query_relation"):
        return self.con.sql(query, alias)

    def execute(self, query, parameters = None, multiple_parameter_sets = False):
        return self.con.execute(query, parameters, multiple_parameter_sets)

    def executemany(self, query, parameters = None):
        return self.con.executemany(query, parameters)

    def close(self):
        return self.con.close()

    def get_csr_v(self, csr_id):
        query = "SELECT csrv FROM sqlpgq_get_csr_v({});".format(csr_id)
        return self.sql(query).fetchnumpy()['csrv'].data

    def get_csr_e(self, csr_id):
        query = "SELECT csre FROM sqlpgq_get_csr_e({});".format(csr_id)
        return self.sql(query).fetchnumpy()['csre'].data

    def get_csr_w(self, csr_id):
        # TODO: we need a scala udf to check if the w is int or double
        #       now, we just only use int w
        query = "SELECT csrw FROM sqlpgq_get_csr_w({});".format(csr_id)
        return self.sql(query).fetchnumpy()['csrw'].data

    def get_csr(self, csr_id):
        v = self.get_csr_v(csr_id)
        e = self.get_csr_e(csr_id)
        w = self.get_csr_w(csr_id)
        return CSR(v, e, w)

    def get_vtablenames(self, pgname):
        query = "SELECT vtables FROM sqlpgq_get_pg_vtablenames({});".format(pgname)
        return self.sql(query).fetchnumpy()["vtables"].data

    def get_etablenames(self, pgname):
        query = "SELECT etables FROM sqlpgq_get_pg_etablenames({});".format(pgname)
        return self.sql(query).fetchnumpy()["etables"].data

    def get_vcolnames(self, pgname, table_name):
        query = "SELECT colnames FROM sqlpgq_get_pg_vcolnames({}, {});".format(pgname, table_name)
        return self.sql(query).fetchnumpy()["colnames"].data

    def get_ecolnames(self, pgname, table_name):
        query = "SELECT colnames FROM sqlpgq_get_pg_ecolnames({}, {});".format(pgname, table_name)
        return self.sql(query).fetchnumpy()["colnames"].data

    def get_edge_features(self, pgname):
        etables = self.get_etablenames(pgname)
        result = {}
        for table in etables:
            col_names = self.get_ecolnames(pgname, table)
            query = _gen_query(table, col_names)
            result[table] = self.sql(query).fetchnumpy()
        return result

    def get_node_features(self, pgname):
        vtables = self.get_vtablenames(pgname)
        result = {}
        for table in vtables:
            col_names = self.get_vcolnames(pgname, table)
            query = _gen_query(table, col_names)
            result[table] = self.sql(query).fetchnumpy()
        return result

    def get_pg(self, pgname, csr_id): # TODO: csr_id optional
        nodes = self.get_node_features(pgname)
        edges = self.get_edge_features(pgname)
        csr = self.get_csr(csr_id)

        return IntermediateGraph(csr, edges, nodes)

    def get_hetero_pg(self, pgname, csr_ids):
        nodes = self.get_node_features(pgname)
        edges = self.get_edge_features(pgname)
        csrs = []
        for elem in csr_ids:
            if len(elem) == 3:
                csrs.append( ( elem[0], self.get_csr(elem[1]), elem[2] ) )
            elif len(elem) == 1:
                csrs.append( self.get_csr(elem[0]) )
        return IntermediateGraph(csrs, edges, nodes)

    def create_property_graph(self, name, nodes, edges):
        query = "CREATE PROPERTY GRAPH {} ".format(name)
        query += "VERTEX TABLES ("
        if type(nodes) == VertexTable:
            query += nodes._to_query_expr()
        else:
            temp = ""
            for elem in nodes:
                if (type(elem) != VertexTable):
                    raise Exception("Not a VertexTable")
                temp += elem._to_query_expr()
                temp += ", "
            temp.rstrip(", ") + " "

        query += ") EDGE TABLES ("
        if type(edges) == EdgeTable:
            query += edges._to_query_expr()
        else:
            temp = ""
            for elem in edges:
                if (type(elem) != EdgeTable):
                    raise Exception("Not a EdgeTable")
                temp += elem._to_query_expr()
                temp += ", "
            temp.rstrip(", ") + " "

        query += ");"

        self.sql(query)

    def _create_csr(self, csr_id, table_name, src_key, dst_key, src_ref_table, src_ref_col, dst_ref_table, dst_ref_col):
        query = """
        SELECT  CREATE_CSR_EDGE(
            {csr_id},
            (SELECT count(srctable.{src_ref_col}) FROM {src_ref_table} srctable),
            CAST (
                (SELECT sum(CREATE_CSR_VERTEX(
                            {csr_id},
                            (SELECT count(srctable.{src_ref_col}) FROM {src_ref_table} srctable),
                            sub.dense_id,
                            sub.cnt)
                            )
                FROM (
                    SELECT srctable.rowid as dense_id, count(edgetable.{src_key}) as cnt
                    FROM {src_ref_table} srctable
                    LEFT JOIN {edge_table} edgetable ON edgetable.{src_key} = srctable.{src_ref_col}
                    GROUP BY srctable.rowid) sub
                )
            AS BIGINT),
            srctable.rowid,
            dsttable.rowid,
            edgetable.rowid) as temp
    FROM {edge_table} edgetable
    JOIN {src_ref_table} srctable on srctable.{src_ref_col} = edgetable.{src_key}
    JOIN {dst_ref_table} dsttable on dsttable.{dst_ref_col} = edgetable.{dst_key} ;""".format(csr_id = csr_id, src_ref_table = src_ref_table, src_ref_col = src_ref_col, 
                                                                                               edge_table = table_name, src_key = src_key, dst_key = dst_key,
                                                                                               dst_ref_table = dst_ref_table, dst_ref_col = dst_ref_col)
        return self.sql(query).fetchnumpy()

    def create_csr(self,
                   pg = None,
                   edge_table = None,
                   table_name = None, src_key = None, dst_key = None, src_ref_table = None, src_ref_col = None, dst_ref_table = None, dst_ref_col = None, csr_id = None):
        if csr_id == None:
            csr_id = 1
        if pg:
            raise Exception("Build CSR from {} is not supported yet".format(type(pg)))
        elif edge_table:
            if type(edge_table) != EdgeTable:
                raise Exception("invalid parameters")
            table_name = edge_table.table_name
            src_key = edge_table.src_key
            dst_key = edge_table.dst_key
            src_ref_table = edge_table.src_ref_table
            src_ref_col = edge_table.src_ref_col
            dst_ref_table = edge_table.dst_ref_table
            dst_ref_col = edge_table.dst_ref_col
            self._create_csr(csr_id, table_name, src_key, dst_key, src_ref_table, src_ref_col, dst_ref_table, dst_ref_col)
        elif table_name and src_key and dst_key and src_ref_table and src_ref_col and dst_ref_table and dst_ref_col:
            self._create_csr(csr_id, table_name, src_key, dst_key, src_ref_table, src_ref_col, dst_ref_table, dst_ref_col)
        else:
            raise Exception("invalid parameters")

    def save_pg(self, graph):
        """ this will require some memory copies. because we will have a create tables
        """
        pass
