/* Python embed for thu-ml SageAttention3 Blackwell FP4.
 * Compiled as build-cuda/libh3_sage3.so and dlopened by h3_gpu_cuda.cu
 * (--sdpa-backend sage3). Mirrors libh3_sol_cute.so ABI. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static PyObject *g_mod = NULL;
static PyObject *g_fwd = NULL;

static void set_err(char *error, size_t error_size, const char *msg);
static void set_py_err(char *error, size_t error_size, const char *prefix);

static int ensure_mod(const char *pkg_root, char *error, size_t error_size) {
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            set_err(error, error_size, "Py_Initialize failed");
            return 0;
        }
    }
    if (pkg_root && pkg_root[0]) {
        PyObject *sys = PyImport_ImportModule("sys");
        PyObject *path = sys ? PyObject_GetAttrString(sys, "path") : NULL;
        if (path) {
            PyObject *p = PyUnicode_FromString(pkg_root);
            if (p) {
                PyList_Insert(path, 0, p);
                Py_DECREF(p);
            }
            Py_DECREF(path);
        }
        Py_XDECREF(sys);
        PyErr_Clear();
    }
    if (g_mod) return 1;
    g_mod = PyImport_ImportModule("h3_sage3_embed");
    if (!g_mod) {
        PyObject *sys = PyImport_ImportModule("sys");
        PyObject *path = sys ? PyObject_GetAttrString(sys, "path") : NULL;
        if (path) {
            const char *tools = getenv("H3_SAGE3_TOOLS");
            if (tools && tools[0]) {
                PyObject *p = PyUnicode_FromString(tools);
                if (p) {
                    PyList_Insert(path, 0, p);
                    Py_DECREF(p);
                }
            }
            Py_DECREF(path);
        }
        Py_XDECREF(sys);
        PyErr_Clear();
        g_mod = PyImport_ImportModule("h3_sage3_embed");
    }
    if (!g_mod) {
        set_py_err(error, error_size, "import h3_sage3_embed");
        return 0;
    }
    return 1;
}

static void set_err(char *error, size_t error_size, const char *msg) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s", msg ? msg : "sage3 error");
}

static void set_py_err(char *error, size_t error_size, const char *prefix) {
    PyObject *type = NULL, *value = NULL, *tb = NULL;
    const char *text = "python error";
    PyErr_Fetch(&type, &value, &tb);
    PyErr_NormalizeException(&type, &value, &tb);
    if (value) {
        PyObject *s = PyObject_Str(value);
        if (s) {
            text = PyUnicode_AsUTF8(s);
            snprintf(error, error_size, "%s: %s", prefix, text ? text : "?");
            Py_DECREF(s);
            Py_XDECREF(type);
            Py_XDECREF(value);
            Py_XDECREF(tb);
            return;
        }
    }
    snprintf(error, error_size, "%s: %s", prefix, text);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(tb);
}

int h3_sage3_init(const char *pkg_root, int seq, int heads, int dim,
                  float scale, int unused0, int unused1, void **q_ptr,
                  void **k_ptr, void **v_ptr, void **o_ptr, char *backend,
                  size_t backend_n, char *error, size_t error_size) {
    PyObject *fn = NULL, *args = NULL, *ret = NULL;
    unsigned long long qp, kp, vp, op;
    const char *be;

    (void)unused0;
    (void)unused1;
    if (!q_ptr || !k_ptr || !v_ptr || !o_ptr) {
        set_err(error, error_size, "sage3 init null ptrs");
        return 0;
    }
    if (!ensure_mod(pkg_root, error, error_size)) return 0;
    fn = PyObject_GetAttrString(g_mod, "init");
    if (!fn) {
        set_py_err(error, error_size, "embed.init");
        return 0;
    }
    args = Py_BuildValue("(siiifii)", pkg_root ? pkg_root : "", seq, heads, dim,
                         (double)scale, 0, 0);
    ret = PyObject_CallObject(fn, args);
    Py_DECREF(fn);
    Py_DECREF(args);
    if (!ret) {
        set_py_err(error, error_size, "embed.init()");
        return 0;
    }
    if (!PyArg_ParseTuple(ret, "KKKKs", &qp, &kp, &vp, &op, &be)) {
        Py_DECREF(ret);
        set_err(error, error_size, "embed.init bad return");
        return 0;
    }
    *q_ptr = (void *)(uintptr_t)qp;
    *k_ptr = (void *)(uintptr_t)kp;
    *v_ptr = (void *)(uintptr_t)vp;
    *o_ptr = (void *)(uintptr_t)op;
    if (backend && backend_n)
        snprintf(backend, backend_n, "%s", be ? be : "sage3");
    g_fwd = PyObject_GetAttrString(g_mod, "fwd");
    Py_DECREF(ret);
    if (!g_fwd) {
        set_py_err(error, error_size, "embed.fwd");
        return 0;
    }
    return 1;
}

int h3_sage3_fwd_ex(char *error, size_t error_size, double *prep_ms,
                    double *kern_ms, double *copy_ms, double *compile_ms,
                    double *host_ms) {
    PyObject *ret;
    if (prep_ms) *prep_ms = 0.0;
    if (kern_ms) *kern_ms = 0.0;
    if (copy_ms) *copy_ms = 0.0;
    if (compile_ms) *compile_ms = 0.0;
    if (host_ms) *host_ms = 0.0;
    if (!g_fwd) {
        set_err(error, error_size, "sage3 not inited");
        return 0;
    }
    ret = PyObject_CallObject(g_fwd, NULL);
    if (!ret) {
        set_py_err(error, error_size, "embed.fwd()");
        return 0;
    }
    if (PyTuple_Check(ret)) {
        double okd = 1.0, prep = 0.0, kern = 0.0, copy = 0.0, compile = 0.0,
               host = 0.0;
        if (!PyArg_ParseTuple(ret, "dddddd", &okd, &prep, &kern, &copy,
                              &compile, &host)) {
            PyErr_Clear();
            Py_DECREF(ret);
            set_err(error, error_size, "embed.fwd bad tuple");
            return 0;
        }
        if (prep_ms) *prep_ms = prep;
        if (kern_ms) *kern_ms = kern;
        if (copy_ms) *copy_ms = copy;
        if (compile_ms) *compile_ms = compile;
        if (host_ms) *host_ms = host;
        Py_DECREF(ret);
        return okd != 0.0;
    }
    Py_DECREF(ret);
    return 1;
}

int h3_sage3_fwd(char *error, size_t error_size) {
    return h3_sage3_fwd_ex(error, error_size, NULL, NULL, NULL, NULL, NULL);
}

int h3_sage3_warmup(const char *pkg_root, int seq, int heads, int dim,
                    float scale, int unused0, int unused1, char *error,
                    size_t error_size) {
    PyObject *fn = NULL, *args = NULL, *ret = NULL;
    (void)unused0;
    (void)unused1;
    if (!ensure_mod(pkg_root, error, error_size)) return 0;
    fn = PyObject_GetAttrString(g_mod, "warmup");
    if (!fn) {
        set_py_err(error, error_size, "embed.warmup");
        return 0;
    }
    args = Py_BuildValue("(siiifii)", pkg_root ? pkg_root : "", seq, heads, dim,
                         (double)scale, 0, 0);
    ret = PyObject_CallObject(fn, args);
    Py_DECREF(fn);
    Py_DECREF(args);
    if (!ret) {
        set_py_err(error, error_size, "embed.warmup()");
        return 0;
    }
    Py_DECREF(ret);
    return 1;
}
