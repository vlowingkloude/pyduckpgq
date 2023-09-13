#include <pybind11/pybind11.h>
#include <atomic>
#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;

using namespace std;

py::list GetCSR(uint64_t v, uint64_t e, uint64_t w, uint64_t vsize, uint64_t wtype) {
	py::list res;

        atomic<int64_t> * temp = (atomic<int64_t> *)v;
        auto capsule = py::capsule(reinterpret_cast<int64_t *>(temp), [](void *csre) {});
        res.append(py::array_t<int64_t>{(int64_t)vsize, reinterpret_cast<int64_t *>(temp), capsule});

        vector<int64_t> *csre = ((vector<int64_t> *)e);
        capsule = py::capsule(csre, [](void *csre) {});
        res.append(py::array(csre->size(), csre->data(), capsule));

	if(wtype == 0) {
		vector<int64_t> *csrw = ((vector<int64_t> *)w);
		capsule = py::capsule(csrw, [](void *csrw) {});
		res.append(py::array(csrw->size(), csrw->data(), capsule));
	} else if(wtype == 1) {
		vector<double> *csrw = ((vector<double> *)w);
		capsule = py::capsule(csrw, [](void *csrw) {});
		res.append(py::array(csrw->size(), csrw->data(), capsule));
	} else {
		// better return an empty array
		// so that always three arrays will be returned.
		res.append(py::array());
	}
        return res;
}

PYBIND11_MODULE(duckcsr, m) {
    m.def("get_csr", &GetCSR, "Get CSR from DuckPGQ via zero-copy");
}
