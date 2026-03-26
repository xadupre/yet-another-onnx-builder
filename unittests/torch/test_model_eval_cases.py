import unittest
from yobx.ext_test_case import ExtTestCase, requires_torch, ignore_warnings
from yobx.torch.model_eval_cases import discover, evaluation


class TestModelEvalCases(ExtTestCase):
    @requires_torch("2.7", "scan")
    @ignore_warnings(FutureWarning)
    def test_discover(self):
        res = discover()
        self.assertNotEmpty(res)
        for mod in res.values():
            with self.subTest(name=mod.__name__):
                if mod.__name__ == "ControlFlowCondIdentity_153832":
                    raise unittest.SkipTest("ControlFlowCondIdentity_153832 needs missing clone.")
                m = mod()
                if isinstance(m._inputs, tuple):
                    m(*m._inputs)
                else:
                    for v in m._inputs:
                        m(*v)
                if hasattr(m, "_valid"):
                    for v in m._valid:
                        m(*v)

    def test_eval(self):
        d = list(discover().items())[0]  # noqa: RUF015
        ev = evaluation(
            quiet=False,
            cases={d[0]: d[1]},
            exporters=(
                "export-strict",
                "export-nostrict",
                "custom",
                "dynamo",
                "dynamo-ir",
                "export-tracing",
                "yobx",
                "yobx-tracing",
            ),
        )
        self.assertIsInstance(ev, list)
        self.assertIsInstance(ev[0], dict)

    def test_run_exporter_custom(self):
        evaluation(
            cases="SignatureListFixedLength", exporters="custom", quiet=False, dynamic=False
        )

    def test_run_exporter_yobx(self):
        evaluation(cases="SignatureListFixedLength", exporters="yobx", quiet=False, dynamic=False)

    def test_run_exporter_dynamo(self):
        evaluation(
            cases="SignatureListFixedLength", exporters="dynamo", quiet=False, dynamic=False
        )

    def test_run_exporter_dynamo_ir(self):
        evaluation(
            cases="SignatureListFixedLength", exporters="dynamo-ir", quiet=False, dynamic=False
        )

    def test_run_exporter_nostrict(self):
        evaluation(
            cases="SignatureListFixedLength",
            exporters="export-nostrict",
            quiet=False,
            dynamic=False,
        )

    def test_run_exporter_tracing(self):
        evaluation(
            cases="SignatureListFixedLength",
            exporters="export-tracing",
            quiet=False,
            dynamic=False,
        )

    def test_run_exporter_yobx_tracing(self):
        evaluation(
            cases="SignatureListFixedLength", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_regex(self):
        evaluation(cases=".*Aten.*", exporters="custom-strict", quiet=False, dynamic=False)

    def test_run_exporter_custom_nested_cond(self):
        evaluation(cases="ControlFlowNestCond", exporters="custom", quiet=False, dynamic=False)

    def test_run_exporter_yobx_tracing_cond(self):
        evaluation(cases="ControlFlowCond", exporters="yobx-tracing", quiet=False, dynamic=False)

    def test_run_exporter_yobx_tracing_cond_2outputs(self):
        evaluation(
            cases="ControlFlowCond2Outputs", exporters="yobx-tracing", quiet=False, dynamic=False
        )

    def test_run_exporter_yobx_tracing_controlflow_rank(self):
        evaluation(
            cases="ControlFlowRanks", exporters="yobx-tracing", quiet=False, dynamic=False
        )

    def test_run_exporter_crop_last_dim_tensor_content(self):
        evaluation(
            cases="CropLastDimensionWithTensorContent",
            exporters="custom",
            quiet=False,
            dynamic=True,
        )

    def test_run_exporter_dimension0(self):
        evaluation(
            cases="ExportWithDimension0",
            exporters="export-nostrict-oblivious",
            quiet=False,
            dynamic=True,
        )

    def test_run_exporter_dimension0_tracing(self):
        evaluation(
            cases="ExportWithDimension0", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_dimension1(self):
        evaluation(
            cases="ExportWithDimension1",
            exporters="export-nostrict-oblivious",
            quiet=False,
            dynamic=True,
        )

    @requires_torch("2.7", "scan")
    def test_run_exporter_vmap_python_nostrict(self):
        evaluation(cases="VmapPython", exporters="export-nostrict", quiet=False, dynamic=True)

    @requires_torch("2.7", "scan")
    def test_run_exporter_vmap_python_yobx(self):
        evaluation(cases="VmapPython", exporters="yobx", quiet=False, dynamic=True)

    def test_run_exporter_vmap_tracing(self):
        evaluation(cases="Vmap", exporters="yobx-tracing", quiet=False, dynamic=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
