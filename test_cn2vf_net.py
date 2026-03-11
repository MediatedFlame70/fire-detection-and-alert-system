"""
Comprehensive test suite for CN2VF-Net model.
Tests model architecture, forward pass, output shapes, and component integrity.
"""

import torch
import torch.nn as nn
from cn2vf_net import (
    CN2VFNet,
    ConvBNAct,
    InvertedResidual,
    PatchEmbed,
    MHSABlock,
    TransformerStage,
    TokenDownsample,
    MultiScaleFusion,
    DetectionHead,
)


def test_model_instantiation():
    """Test 1: Model can be instantiated without errors."""
    print("Test 1: Model Instantiation...", end=" ")
    try:
        model = CN2VFNet(num_classes=3)
        assert isinstance(model, nn.Module)
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_forward_pass_shapes():
    """Test 2: Forward pass with correct input shape produces expected outputs."""
    print("Test 2: Forward Pass with Shape [2, 3, 448, 448]...", end=" ")
    try:
        model = CN2VFNet(num_classes=3)
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(2, 3, 448, 448)
            output = model(x)
            
            assert "cls_logits" in output, "Missing cls_logits in output"
            assert "bbox" in output, "Missing bbox in output"
            assert output["cls_logits"].shape == (2, 3), f"Expected (2, 3), got {output['cls_logits'].shape}"
            assert output["bbox"].shape == (2, 4), f"Expected (2, 4), got {output['bbox'].shape}"
            
            # Verify bbox is normalized [0, 1]
            assert torch.all(output["bbox"] >= 0) and torch.all(output["bbox"] <= 1), "BBox values not in [0, 1]"
            
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_batch_size_variations():
    """Test 3: Model handles different batch sizes."""
    print("Test 3: Different Batch Sizes [1, 4, 8]...", end=" ")
    try:
        model = CN2VFNet(num_classes=3)
        model.eval()
        
        for batch_size in [1, 4, 8]:
            with torch.no_grad():
                x = torch.randn(batch_size, 3, 448, 448)
                output = model(x)
                assert output["cls_logits"].shape == (batch_size, 3)
                assert output["bbox"].shape == (batch_size, 4)
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_gradient_flow():
    """Test 4: Gradients can flow backward through the model."""
    print("Test 4: Gradient Flow (Backward Pass)...", end=" ")
    try:
        model = CN2VFNet(num_classes=3)
        model.train()
        
        x = torch.randn(2, 3, 448, 448, requires_grad=True)
        output = model(x)
        
        # Compute dummy loss
        loss = output["cls_logits"].sum() + output["bbox"].sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None, "Input gradient is None"
        
        # Check some model parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No model parameters have gradients"
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_component_conv_bn_act():
    """Test 5: ConvBNAct component."""
    print("Test 5: ConvBNAct Component...", end=" ")
    try:
        layer = ConvBNAct(16, 32, kernel_size=3, stride=2)
        x = torch.randn(2, 16, 56, 56)
        out = layer(x)
        assert out.shape == (2, 32, 28, 28), f"Expected (2, 32, 28, 28), got {out.shape}"
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_component_inverted_residual():
    """Test 6: InvertedResidual block."""
    print("Test 6: InvertedResidual Block...", end=" ")
    try:
        # Test with residual connection
        block1 = InvertedResidual(32, 32, stride=1, expand_ratio=4.0)
        x1 = torch.randn(2, 32, 56, 56)
        out1 = block1(x1)
        assert out1.shape == (2, 32, 56, 56), f"Expected (2, 32, 56, 56), got {out1.shape}"
        
        # Test without residual (stride=2)
        block2 = InvertedResidual(32, 64, stride=2, expand_ratio=4.0)
        x2 = torch.randn(2, 32, 56, 56)
        out2 = block2(x2)
        assert out2.shape == (2, 64, 28, 28), f"Expected (2, 64, 28, 28), got {out2.shape}"
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_component_patch_embed():
    """Test 7: PatchEmbed converts 2D to 1D tokens."""
    print("Test 7: PatchEmbed Component...", end=" ")
    try:
        patch_embed = PatchEmbed(in_ch=80, embed_dim=128)
        x = torch.randn(2, 80, 28, 28)
        tokens, h, w = patch_embed(x)
        
        assert tokens.shape == (2, 28 * 28, 128), f"Expected (2, 784, 128), got {tokens.shape}"
        assert h == 28 and w == 28, f"Expected h=28, w=28, got h={h}, w={w}"
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_component_mhsa_block():
    """Test 8: MHSA Transformer block."""
    print("Test 8: MHSA Transformer Block...", end=" ")
    try:
        block = MHSABlock(dim=128, num_heads=4, mlp_ratio=4.0)
        x = torch.randn(2, 784, 128)
        out = block(x)
        assert out.shape == (2, 784, 128), f"Expected (2, 784, 128), got {out.shape}"
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_component_transformer_stage():
    """Test 9: TransformerStage with multiple blocks."""
    print("Test 9: TransformerStage (depth=2)...", end=" ")
    try:
        stage = TransformerStage(dim=128, depth=2, num_heads=4, mlp_ratio=4.0)
        x = torch.randn(2, 784, 128)
        out = stage(x)
        assert out.shape == (2, 784, 128), f"Expected (2, 784, 128), got {out.shape}"
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_component_token_downsample():
    """Test 10: TokenDownsample reduces spatial resolution."""
    print("Test 10: TokenDownsample (28x28 → 14x14)...", end=" ")
    try:
        downsample = TokenDownsample(in_dim=128, out_dim=160)
        x = torch.randn(2, 784, 128)  # 28*28 = 784
        out, h_out, w_out = downsample(x, h=28, w=28)
        
        assert out.shape == (2, 196, 160), f"Expected (2, 196, 160), got {out.shape}"
        assert h_out == 14 and w_out == 14, f"Expected h=14, w=14, got h={h_out}, w={w_out}"
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_component_multiscale_fusion():
    """Test 11: MultiScaleFusion merges features."""
    print("Test 11: MultiScaleFusion...", end=" ")
    try:
        fusion = MultiScaleFusion(c2_dim=40, c3_dim=80, t_dim=160, fuse_dim=128)
        
        c2 = torch.randn(2, 40, 56, 56)
        c3 = torch.randn(2, 80, 28, 28)
        t2_map = torch.randn(2, 160, 14, 14)
        
        fused = fusion(c2, c3, t2_map)
        assert fused.shape == (2, 128, 56, 56), f"Expected (2, 128, 56, 56), got {fused.shape}"
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_component_detection_head():
    """Test 12: DetectionHead outputs class and bbox."""
    print("Test 12: DetectionHead...", end=" ")
    try:
        head = DetectionHead(in_dim=128, num_classes=3)
        x = torch.randn(2, 128, 56, 56)
        cls_logits, bbox = head(x)
        
        assert cls_logits.shape == (2, 3), f"Expected (2, 3), got {cls_logits.shape}"
        assert bbox.shape == (2, 4), f"Expected (2, 4), got {bbox.shape}"
        assert torch.all(bbox >= 0) and torch.all(bbox <= 1), "BBox not normalized"
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_model_parameter_count():
    """Test 13: Model has reasonable parameter count."""
    print("Test 13: Parameter Count...", end=" ")
    try:
        model = CN2VFNet(num_classes=3)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ PASSED (Total: {total_params:,}, Trainable: {trainable_params:,})")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_model_inference_mode():
    """Test 14: Model works in eval mode (no dropout/batchnorm issues)."""
    print("Test 14: Inference Mode Consistency...", end=" ")
    try:
        model = CN2VFNet(num_classes=3)
        model.eval()
        
        x = torch.randn(1, 3, 448, 448)
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
            
            # Same input should give same output in eval mode
            assert torch.allclose(out1["cls_logits"], out2["cls_logits"], atol=1e-6)
            assert torch.allclose(out1["bbox"], out2["bbox"], atol=1e-6)
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_model_output_ranges():
    """Test 15: Output ranges are valid."""
    print("Test 15: Output Value Ranges...", end=" ")
    try:
        model = CN2VFNet(num_classes=3)
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(4, 3, 448, 448)
            output = model(x)
            
            # cls_logits can be any real number (logits)
            assert torch.isfinite(output["cls_logits"]).all(), "cls_logits contains NaN/Inf"
            
            # bbox should be in [0, 1] due to sigmoid
            assert (output["bbox"] >= 0).all() and (output["bbox"] <= 1).all(), "bbox out of [0,1]"
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def run_all_tests():
    """Run all test cases and report results."""
    print("=" * 70)
    print("CN2VF-Net Model Test Suite")
    print("=" * 70)
    print()
    
    tests = [
        test_model_instantiation,
        test_forward_pass_shapes,
        test_batch_size_variations,
        test_gradient_flow,
        test_component_conv_bn_act,
        test_component_inverted_residual,
        test_component_patch_embed,
        test_component_mhsa_block,
        test_component_transformer_stage,
        test_component_token_downsample,
        test_component_multiscale_fusion,
        test_component_detection_head,
        test_model_parameter_count,
        test_model_inference_mode,
        test_model_output_ranges,
    ]
    
    results = []
    for test_func in tests:
        passed = test_func()
        results.append(passed)
    
    print()
    print("=" * 70)
    passed_count = sum(results)
    total_count = len(results)
    print(f"Test Results: {passed_count}/{total_count} PASSED")
    
    if passed_count == total_count:
        print("🎉 All tests passed successfully!")
    else:
        print(f"⚠️  {total_count - passed_count} test(s) failed.")
    
    print("=" * 70)
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
