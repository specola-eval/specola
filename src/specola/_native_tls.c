/* Existing module TLS lookup without Python callbacks under the loader lock.
 *
 * The Python owner authenticates the public function address and the expected
 * image before entry. This bridge does not resolve symbols, initialize TLS,
 * walk private loader storage, or release the GIL. See the dated TLS design
 * for the pinned glibc and CPython source contracts and their limits.
 */
#define PY_SSIZE_T_CLEAN
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <Python.h>
#include <link.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>

typedef int (*specola_phdr_iterator)(
    int (*)(struct dl_phdr_info *, size_t, void *), void *);

struct specola_tls_lookup {
    uintptr_t image;
    uintptr_t phdr;
    size_t phnum;
    size_t matches;
    int short_record;
    uintptr_t tls;
    size_t module_id;
};

static int specola_tls_callback(struct dl_phdr_info *info, size_t size,
                                void *opaque) {
    struct specola_tls_lookup *lookup = opaque;
    /* The public record can grow; its supplied size owns the read boundary. */
    if (info == NULL || size < offsetof(struct dl_phdr_info, dlpi_tls_data)
                              + sizeof(info->dlpi_tls_data)) {
        lookup->short_record = 1;
        return 1;
    }
    if ((uintptr_t)info->dlpi_addr == lookup->image &&
        (uintptr_t)info->dlpi_phdr == lookup->phdr &&
        (size_t)info->dlpi_phnum == lookup->phnum) {
        /* Two matches suffice to reject ambiguity; never wrap a counter. */
        if (lookup->matches != 0) {
            lookup->matches = 2;
            return 1;
        }
        lookup->matches = 1;
        lookup->tls = (uintptr_t)info->dlpi_tls_data;
        lookup->module_id = info->dlpi_tls_modid;
    }
    return 0;
}

static PyObject *specola_existing_tls(PyObject *self, PyObject *args) {
    PyObject *objects[4];
    unsigned long long values[4];
    struct specola_tls_lookup lookup = {0};
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOO:existing_tls", &objects[0], &objects[1],
                          &objects[2], &objects[3])) {
        return NULL;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (!PyLong_CheckExact(objects[i])) {
            PyErr_SetString(PyExc_TypeError, "TLS lookup requires exact integers");
            return NULL;
        }
        values[i] = PyLong_AsUnsignedLongLong(objects[i]);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    if (sizeof(uintptr_t) != sizeof(ElfW(Addr)) ||
        values[0] == 0 || values[0] > UINTPTR_MAX ||
        values[1] > UINTPTR_MAX || values[2] == 0 || values[2] > UINTPTR_MAX ||
        values[3] == 0 || values[3] > (unsigned long long)((ElfW(Half))-1)) {
        PyErr_SetString(PyExc_ValueError, "TLS lookup address or ABI is invalid");
        return NULL;
    }
    lookup.image = (uintptr_t)values[1];
    lookup.phdr = (uintptr_t)values[2];
    lookup.phnum = (size_t)values[3];
    specola_phdr_iterator iterate = (specola_phdr_iterator)(uintptr_t)values[0];
    /* No Python API, allocation or GIL release until iterate has unlocked. */
    int result = iterate(specola_tls_callback, &lookup);
    if (lookup.short_record) {
        PyErr_SetString(PyExc_RuntimeError, "native TLS callback record is short");
        return NULL;
    }
    if (result != 0 || lookup.matches != 1) {
        PyErr_SetString(PyExc_RuntimeError, "native TLS image is absent or ambiguous");
        return NULL;
    }
    return Py_BuildValue("KK", (unsigned long long)lookup.module_id,
                          (unsigned long long)lookup.tls);
}

static PyObject *specola_tls_layout(PyObject *self, PyObject *unused) {
    (void)self;
    (void)unused;
#define SPECOLA_FIELD_LAYOUT(name) \
    (unsigned long long)offsetof(struct dl_phdr_info, name), \
    (unsigned long long)sizeof(((struct dl_phdr_info *)0)->name)
    return Py_BuildValue("(KKK(KK)(KK)(KK)(KK)(KK)(KK)(KK)(KK))",
        (unsigned long long)sizeof(void *),
        (unsigned long long)sizeof(size_t),
        (unsigned long long)sizeof(struct dl_phdr_info),
        SPECOLA_FIELD_LAYOUT(dlpi_addr), SPECOLA_FIELD_LAYOUT(dlpi_name),
        SPECOLA_FIELD_LAYOUT(dlpi_phdr), SPECOLA_FIELD_LAYOUT(dlpi_phnum),
        SPECOLA_FIELD_LAYOUT(dlpi_adds), SPECOLA_FIELD_LAYOUT(dlpi_subs),
        SPECOLA_FIELD_LAYOUT(dlpi_tls_modid), SPECOLA_FIELD_LAYOUT(dlpi_tls_data));
#undef SPECOLA_FIELD_LAYOUT
}

static PyMethodDef specola_tls_methods[] = {
    {"existing_tls", specola_existing_tls, METH_VARARGS,
     "Return existing TLS identity through an authenticated public iterator."},
    {"layout", specola_tls_layout, METH_NOARGS,
     "Return compiled public record offsets and widths for source reconciliation."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef specola_tls_module = {
    PyModuleDef_HEAD_INIT, "_native_tls", NULL, -1, specola_tls_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__native_tls(void) {
    return PyModule_Create(&specola_tls_module);
}
