import duckdb
import numpy as np
import decorators

class CSR:
    def __init__(self, v, e, w):
        self.v = v
        self.e = e
        self.w = w

def connect(path = ":memory:", read_only = False, config = {}):
    return Connection(path, read_only, config)

class Connection:
    """ A wrapper for DuckDB connections
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

    def _get_csr_v(self, csr_id):
        query = "SELECT csrv FROM get_csr_v({});".format(csr_id)
        return self.sql(query).torch()['csrv']

    def _get_csr_e(self, csr_id):
        query = "SELECT csre FROM get_csr_e({});".format(csr_id)
        return self.sql(query).torch()['csre']

    def _get_csr_w(self, csr_id):
        query = f"SELECT csr_get_w_type(csr_id)"
        # TODO: call different table functions based on weight type.
        query = "SELECT csrw FROM get_csr_w({});".format(csr_id)
        return self.sql(query).torch()['csrw']

    def get_csr(self, csr_id):
        v = self._get_csr_v(csr_id)
        e = self._get_csr_e(csr_id)
        w = self._get_csr_w(csr_id)
        return CSR(v,e,w)

    def _get_vtablenames(self, pgname):
        query = "SELECT vtables FROM get_pg_vtablenames({});".format(pgname)
        return self.sql(query).fetchall()#numpy()["vtables"].data

    def _get_etablenames(self, pgname):
        query = "SELECT etables FROM get_pg_etablenames({});".format(pgname)
        return self.sql(query).fetchall()#numpy()["etables"].data

    def _get_vcolnames(self, pgname, table_name):
        query = "SELECT colnames FROM get_pg_vcolnames({}, {});".format(pgname, table_name)
        return self.sql(query).fetchnumpy()["colnames"].data

    def _get_ecolnames(self, pgname, table_name):
        query = "SELECT colnames FROM get_pg_ecolnames({}, {});".format(pgname, table_name)
        return self.sql(query).fetchnumpy()["colnames"].data

    def _get_features(self, tablename, cols, drop_non_numeric = True):
        NUMERIC_TYPES = [duckdb.typing.BIGINT, duckdb.typing.DOUBLE, duckdb.typing.TINYINT, duckdb.typing.SMALLINT, duckdb.typing.UBIGINT, duckdb.typing.FLOAT, duckdb.typing.UINTEGER, duckdb.typing.BOOLEAN, duckdb.typing.USMALLINT, duckdb.typing.INTEGER, duckdb.typing.UTINYINT]
        colstr = ", ".join(cols)
        query = f"SELECT {colstr} FROM {tablename};"
        query_result = self.sql(query)
        cols_to_drop = []
        for i in range(len(cols)):
            if query_result.types[i] not in NUMERIC_TYPES:
                if drop_non_numeric:
                    cols_to_drop.append(cols[i])
                else:
                    raise Exception(f"Found a non numeric column {query_result.columns[i]} in table {tablename}")
        if len(cols_to_drop) == 0:
            return query_result.torch()
        else:
            cols = np.setdiff1d(cols, cols_to_drop)
            colstr = ", ".join(cols)
            query = f"SELECT {colstr} FROM {tablename};"
            return self.sql(query).torch()

    def get_pyg(self, pgname, csr_id, drop_non_numeric = True):
        import torch
        from torch_geometric.data import Data
        from torch_geometric.typing import SparseTensor

        vtables = self._get_vtablenames(pgname)
        etables = self._get_etablenames(pgname)
        csr = self.get_csr(csr_id)
        if len(vtables) == 1 and len(etables) == 1:
            if csr.w == None:
                adj_t = SparseTensor(rowptr = csr.v, col = csr.e, value = csr.w)
            else:
                adj_t = SparseTensor(rowptr = csr.v, col = csr.e)
            cols = self._get_vcolnames(pgname, vtables[0][0])
            node_features = self._get_features(vtables[0][0], cols, drop_non_numeric)
            cols = self._get_ecolnames(pgname, etables[0][0])
            edge_features = self._get_features(etables[0][0], cols, drop_non_numeric)
            
            # lets try if there is a huge difference in exec time
            # I came up with two ways.
            # the first is to convert the dict to a 2-dim numpy array first and then cast this numpy array to tensor
            # this method runs faster for smaller data sets.
            # the second if to use torch.stack.
            # this method runs faster for bigger data sets.
            # I decided to use torch.stack
            # first, it leads to shorter and more readable code.
            # second, most of the runing time will be spent on building the SparseTensor.
            # So, especially for smaller data, there wont be some significant gains in performance.
            # For sf = 100, the first will take 27ms while the second will take 19ms.
            # TODO: maybe more bigger data sets?
            """
            newvalues = []
            for i in node_features.values():
                newvalues.append(i.numpy())
            datax = torch.from_numpy(np.array(newvalues))
            newvalues = []
            for i in edge_features.values():
                newvalues.append(i.numpy())
            edge_attr = torch.from_numpy(np.array(newvalues))"""
            datax = torch.stack(list(node_features.values()), dim = 1)
            edge_attr = torch.stack(list(edge_features.values()), dim = 1)
            data = Data(x = datax, edge_attr = edge_attr)
            data.adj_t = adj_t
            return data
        else:
            raise Exception("heterogeneous graph is not supported yet")

    def get_dgl(self, pgname, csr_id, drop_non_numeric = True):
        import dgl
        vtables = self._get_vtablenames(pgname)
        etables = self._get_etablenames(pgname)
        csr = self.get_csr(csr_id)

        if len(vtables) == 1 and len(etables) == 1:
            ## homogeneous graph
            ## TODO: may add a new udf which returns types of each cols
            cols = self._get_vcolnames(pgname, vtables[0][0])
            node_features = self._get_features(vtables[0][0], cols, drop_non_numeric)
            cols = self._get_ecolnames(pgname, etables[0][0])
            edge_features = self._get_features(etables[0][0], cols, drop_non_numeric)
            g = dgl.graph(('csr', (csr.v[:-1], csr.e, [])))
            for k, v in node_features.items():
                g.ndata[k] = v.reshape((v.shape[0], 1))
            for k, v in edge_features.items():
                g.edata[k] = v.reshape((v.shape[0], 1))
            return g
        else:
            ## heterogeneous graph
            raise Exception(f"Call get_dgl_hetero to get heterograph")

    def get_dgl_hetero(self, pgname, csr_dict, drop_non_numeric = True):
        """ csr_dict is a dict.
            keys are csr_ids, values are string triplets (src_type, edge_type, dst_type)
        """
        import dgl
        vtables = self._get_vtablenames(pgname)
        etables = self._get_etablenames(pgname)

        if len(vtables) == 1 and len(etables) == 1:
            raise Exception(f"Call get_dgl_hetero to get homogeneous graph")
        else:
            data_dict = {}
            for k, v in csr_dict.items():
                csr = self.get_csr(k)
                data_dict[v] = ("csr", (csr.v[:-1], csr.e, []))
            g = dgl.heterograph(data_dict)
            for tablename in vtables:
                tablename = tablename[0]
                cols = self._get_vcolnames(pgname, tablename)
                features = self._get_features(tablename, cols, drop_non_numeric)
                for k, v in features.items():
                    g.ndata[tablename + '.' + k] = {tablename: v.reshape((v.shape[0], 1))}
            for tablename in etables:
                tablename = tablename[0]
                cols = self._get_ecolnames(pgname, tablename)
                features = self._get_features(tablename, cols, drop_non_numeric)
                for k, v in features.items():
                    g.edata[tablename + '.' + k] = {tablename: v.reshape((v.shape[0], 1))}
            return g

    def _create_csr(self, csr_id, edge_table, src_key, dst_key, src_ref_table, src_ref_col, dst_ref_table, dst_ref_col):
        """ internal call """
        query = f"""SELECT  CREATE_CSR_EDGE(
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
        JOIN {dst_ref_table} dsttable on dsttable.{dst_ref_col} = edgetable.{dst_key} ;"""
        self.sql(query).fetchnumpy()

    @decorators.require_rowid
    def create_csr(self, edge_table, src_key, dst_key, src_ref_table, src_ref_col, dst_ref_table = None, dst_ref_col = None, csr_id = None):
        if csr_id == None:
            ## TODO: add a new udf to get the a possible csr id.
            csr_id = 1
        ### we allow the dst_ref_table and dst_ref_col to be none
        ### in the cases if src and dst are the same table and col
        if dst_ref_table == None:
            dst_ref_table = src_ref_table
        if dst_ref_col == None:
            dst_ref_col = src_ref_col
        if type(edge_table) == str:
            ##  a duckdb table
            self._create_csr(csr_id, edge_table, src_key, dst_key, src_ref_table, src_ref_col, dst_ref_table, dst_ref_col)
        else:
            ## python objects
            _edgedf = edge_table
            _srcdf = src_ref_table
            _dstdf = dst_ref_table
            self._create_csr(csr_id, "_edgedf", src_key, dst_key, "_srcdf", src_ref_col, "_dstdf", dst_ref_col)
        return csr_id

    def save_dgl_homograph(self, g, node_table_name, edge_feature_table_name, edge_table_name):
        _data_dict = {}
        ## save node features
        for k, v in g.ndata.items():
            _data_dict[k] = v.numpy().reshape(len(v))
        query = "CREATE TABLE {} AS SELECT * FROM _data_dict"
        self.sql(query.format(node_table_name))

        ## save edge features
        _data_dict = {}
        for k, v in g.edata.items():
            _data_dict[k] = v.numpy().reshape(len(v))
        query = "CREATE TABLE {} AS SELECT * FROM _data_dict"
        self.sql(query.format(edge_feature_table_name))

        ## construct edge table
        _data_dict = {}
        _src, _dst = g.adj_tensors("coo")
        _src = _src.numpy()
        _dst = _dst.numpy()
        _data_dict = {"src": _src, "_dst": _dst}
        query = "CREATE TABLE {} AS SELECT * FROM _data_dict"
        self.sql(query.format(edge_table_name))

    def save_dgl_heterograph(self, g):
        pass
