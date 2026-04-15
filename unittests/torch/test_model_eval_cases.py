import unittest
from yobx.ext_test_case import ExtTestCase, requires_torch, ignore_warnings
from yobx.torch.testing.model_eval_cases import discover, evaluation


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

    def test_run_exporter_yobx_new_tracing(self):
        evaluation(
            cases="SignatureListFixedLength",
            exporters="yobx-new-tracing",
            quiet=False,
            dynamic=False,
        )

    def test_run_exporter_regex(self):
        evaluation(cases=".*Aten.*", exporters="custom-strict", quiet=False, dynamic=False)

    def test_run_exporter_custom_nested_cond(self):
        evaluation(cases="ControlFlowNestCond", exporters="custom", quiet=False, dynamic=False)

    def test_run_exporter_yobx_tracing_cond_nested_module(self):
        evaluation(
            cases="ControlFlowCondNestedModule",
            exporters="yobx-tracing",
            quiet=False,
            dynamic=False,
        )

    def test_run_exporter_yobx_tracing_cond(self):
        evaluation(cases="ControlFlowCond", exporters="yobx-tracing", quiet=False, dynamic=False)

    def test_run_exporter_yobx_tracing_cond_2outputs(self):
        evaluation(
            cases="ControlFlowCond2Outputs", exporters="yobx-tracing", quiet=False, dynamic=False
        )

    def test_run_exporter_yobx_tracing_controlflow_rank(self):
        evaluation(cases="ControlFlowRanks", exporters="yobx-tracing", quiet=False, dynamic=False)

    def test_run_exporter_yobx_tracing_controlflow_indirect_rank(self):
        evaluation(
            cases="ControlFlowIndirectRanks", exporters="yobx-tracing", quiet=False, dynamic=False
        )

    def test_run_exporter_yobx_tracing_controlflow_indirect_rank_dynamic(self):
        evaluation(
            cases="ControlFlowIndirectRanks", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_yobx_tracing_controlflow_indirect_rank_cat(self):
        evaluation(
            cases="ControlFlowIndirectRanksCat",
            exporters="yobx-tracing",
            quiet=False,
            dynamic=True,
        )

    def test_run_exporter_yobx_tracing_controlflow_rank_type(self):
        evaluation(
            cases="ControlFlowRanksType", exporters="yobx-tracing", quiet=False, dynamic=False
        )

    def test_run_exporter_crop_last_dim_tensor_content(self):
        evaluation(
            cases="CropLastDimensionWithTensorContent",
            exporters="custom",
            quiet=False,
            dynamic=True,
        )

    def test_run_exporter_crop_last_dim_tensor_content_tracing(self):
        evaluation(
            cases="CropLastDimensionWithTensorContent",
            exporters="yobx-tracing",
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
            cases="ExportWithDimension0",
            exporters="yobx-tracing",
            quiet=False,
            dynamic=True,
            verbose=0,
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

    @requires_torch("2.7", "scan")
    def test_run_exporter_yobx_tracing_scan_cdist(self):
        evaluation(
            cases="ControlFlowScanCDist", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    @requires_torch("2.7", "scan")
    def test_run_exporter_yobx_tracing_scan_cdist2(self):
        evaluation(
            cases="ControlFlowScanCDist2", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_inplace_add2_tracing(self):
        evaluation(cases="InplaceAdd2", exporters="yobx-tracing", quiet=False, dynamic=True)

    def test_run_exporter_inplace_add_mul_tracing(self):
        evaluation(cases="InplaceAdd_Mul", exporters="yobx-tracing", quiet=False, dynamic=True)

    def test_run_exporter_inplace_clone_add_tracing(self):
        evaluation(cases="InplaceCloneAdd_", exporters="yobx-tracing", quiet=False, dynamic=True)

    def test_run_exporter_inplace_setitem_mask_tracing(self):
        evaluation(
            cases="InplaceSetItemMask", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_inplace_setitem_square_tracing(self):
        evaluation(
            cases="InplaceSetItemSquare", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_inplace_setitem_square_add_tracing(self):
        evaluation(
            cases="InplaceSetItemSquareAdd", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_inplace_setitem_square_add2_tracing(self):
        evaluation(
            cases="InplaceSetItemSquareAdd2", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_inplace_setitem_exp_tracing(self):
        evaluation(cases="InplaceSetItemExp", exporters="yobx-tracing", quiet=False, dynamic=True)

    def test_run_exporter_inplace_setitem_ellipsis_1_tracing(self):
        evaluation(
            cases="InplaceSetItemEllipsis_1", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_inplace_setitem_ellipsis_2_tracing(self):
        evaluation(
            cases="InplaceSetItemEllipsis_2", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_aten_as_strided_tracing(self):
        evaluation(cases="AtenAsStrided", exporters="yobx-tracing", quiet=False, dynamic=True)

    def test_run_exporter_aten_as_strided_new_tracing(self):
        evaluation(cases="AtenAsStrided", exporters="yobx-new-tracing", quiet=False, dynamic=True)

    def test_run_exporter_aten_inplace_add_new_tracing(self):
        evaluation(cases="InplaceAdd", exporters="yobx-new-tracing", quiet=False, dynamic=True)

    def test_control_flow_numel_zero_1(self):
        evaluation(
            cases="ControlFlowNumelZero1", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_controlflow_numel_zero_2_tracing(self):
        evaluation(
            cases="ControlFlowNumelZero2", exporters="yobx-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_controlflow_numel_zero_export_1_tracing(self):
        evaluation(
            cases="ControlFlowNumelZero1", exporters="export-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_controlflow_numel_zero_export_2_tracing(self):
        evaluation(
            cases="ControlFlowNumelZero2", exporters="export-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_controlflow_cond_new_tracing(self):
        evaluation(
            cases="ControlFlowCond", exporters="yobx-new-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_controlflow_cond_2outputs_new_tracing(self):
        evaluation(
            cases="ControlFlowCond2Outputs",
            exporters="yobx-new-tracing",
            quiet=False,
            dynamic=True,
        )

    def test_run_exporter_controlflow_shape_check_new_tracing(self):
        evaluation(
            cases="ControlFlowShapeCheck", exporters="yobx-new-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_aten_interpolate_new_tracing(self):
        evaluation(
            cases="AtenInterpolate", exporters="yobx-new-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_aten_roll_pos_new_tracing(self):
        evaluation(cases="AtenRollPos", exporters="yobx-new-tracing", quiet=False, dynamic=True)

    def test_run_exporter_aten_roll_relu_new_tracing(self):
        evaluation(cases="AtenRollRelu", exporters="yobx-new-tracing", quiet=False, dynamic=True)

    @requires_torch("2.7", "scan")
    def test_run_exporter_yobx_scan_new_tracing(self):
        evaluation(
            cases="ControlFlowScan", exporters="yobx-new-tracing", quiet=False, dynamic=True
        )

    @requires_torch("2.7", "scan")
    def test_run_exporter_yobx_scan_cdist_new_tracing(self):
        evaluation(
            cases="ControlFlowScanCDist", exporters="yobx-new-tracing", quiet=False, dynamic=True
        )

    @requires_torch("2.7", "scan")
    def test_run_exporter_yobx_scan_cdist2_new_tracing(self):
        evaluation(
            cases="ControlFlowScanCDist2", exporters="yobx-new-tracing", quiet=False, dynamic=True
        )

    def test_run_exporter_layer_norm_tracing(self):
        evaluation(cases="LayerNorm", exporters="yobx-tracing", quiet=False, dynamic=True)

    def test_run_exporter_layer_norm_new_tracing(self):
        evaluation(cases="LayerNorm", exporters="yobx-new-tracing", quiet=False, dynamic=True)

    def test_run_exporter_builtin_isinstance_new_tracing(self):
        evaluation(
            cases="BuildInIsInstance", exporters="yobx-new-tracing", quiet=False, dynamic=True
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
