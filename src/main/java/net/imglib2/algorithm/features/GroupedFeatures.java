package net.imglib2.algorithm.features;

import net.imglib2.algorithm.features.ops.DifferenceOfGaussiansFeature;
import net.imglib2.algorithm.features.ops.FeatureOp;
import net.imglib2.algorithm.features.ops.SingleSphereShapedFeature;
import net.imglib2.algorithm.features.ops.SobelGradientFeature;
import net.imglib2.algorithm.features.ops.SphereShapedFeature;

import java.util.HashMap;
import java.util.Map;

import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;

/**
 * @author Matthias Arzt
 */
public class GroupedFeatures {

	public static FeatureSetting gabor() {
		return createFeature(net.imglib2.algorithm.features.ops.GaborFeature.class, "legacyNormalize", FALSE);
	}

	public static FeatureSetting legacyGabor() {
		return createFeature(net.imglib2.algorithm.features.ops.GaborFeature.class, "legacyNormalize", TRUE);
	}

	public static FeatureSetting gauss() {
		return createFeature(net.imglib2.algorithm.features.ops.GaussFeature.class);
	}

	public static FeatureSetting sobelGradient() {
		return createFeature(SobelGradientFeature.class);
	}

	public static FeatureSetting gradient() {
		return createFeature(net.imglib2.algorithm.features.ops.GradientFeature.class);
	}

	public static FeatureSetting min() {
		return createSphereShapeFeature(SingleSphereShapedFeature.MIN);
	}

	public static FeatureSetting max() {
		return createSphereShapeFeature(SingleSphereShapedFeature.MAX);
	}

	public static FeatureSetting mean() {
		return createSphereShapeFeature(SingleSphereShapedFeature.MEAN);
	}

	public static FeatureSetting median() {
		return createSphereShapeFeature(SingleSphereShapedFeature.MEDIAN);
	}

	public static FeatureSetting variance() {
		return createSphereShapeFeature(SingleSphereShapedFeature.VARIANCE);
	}

	private static FeatureSetting createSphereShapeFeature(String operation) {
		return createFeature(SphereShapedFeature.class, "operation", operation);
	}

	public static FeatureSetting lipschitz(long border) {
		return createFeature(net.imglib2.algorithm.features.ops.LipschitzFeature.class, "border", border);
	}

	public static FeatureSetting hessian() {
		return createFeature(net.imglib2.algorithm.features.ops.HessianFeature.class);
	}

	public static FeatureSetting differenceOfGaussians() {
		return createFeature(net.imglib2.algorithm.features.ops.DifferenceOfGaussiansFeature.class);
	}

	private static FeatureSetting createFeature(Class<? extends FeatureOp> aClass, Object... args) {
		return new FeatureSetting(aClass, args);
	}
}
