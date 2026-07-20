// PreserveNativeFidelityTests — retired.
//
// This file held the U2c.6 real-model fidelity gate for the preserve_native mechanism
// (native bf16 kept on matched matvec projections instead of an fp16 downcast). Both tests
// were built on a real-model fp16 baseline spec + HF checkpoint that were removed from the
// public tree, so the gate is no longer runnable here.
//
// The mechanism itself remains fully covered by the synthetic-fixture gates (plan U2c.5):
// PreserveNativePatternsTests (layout tagging), PreserveNativeCodegenTests (end-to-end
// build → bf16-W kernel), and PreserveNativeQuantizerTests (storage policy).
