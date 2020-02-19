
package net.imglib2.trainable_segmention.pixel_feature.filter.stats;

import net.imagej.ops.OpEnvironment;
import net.imagej.ops.OpService;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.test.ImgLib2Assert;
import net.imglib2.trainable_segmention.RevampUtils;
import net.imglib2.trainable_segmention.ToString;
import net.imglib2.trainable_segmention.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmention.pixel_feature.filter.FeatureOp;
import net.imglib2.trainable_segmention.pixel_feature.filter.SingleFeatures;
import net.imglib2.trainable_segmention.pixel_feature.settings.FeatureSetting;
import net.imglib2.trainable_segmention.pixel_feature.settings.GlobalSettings;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.junit.Test;
import org.scijava.Context;

public class SingleStatisticsFeatureTest {

	@Test
	public void test() {
		OpEnvironment ops = new Context().service(OpService.class);
		FeatureSetting featureSetting = new FeatureSetting(SingleStatisticsFeature.class, "min", true,
			"max", true, "mean", true, "variance", true, "radius", 1);
		FeatureOp feature = featureSetting.newInstance(ops, GlobalSettings.default2d().build());
		Img<FloatType> input = ArrayImgs.floats(new float[] {
			0, 0, 0, 0,
			0, 1, -1, 0,
			0, 0, 0, 0
		}, 4, 3);
		Img<FloatType> output = ArrayImgs.floats(4, 3, 4);
		Img<FloatType> expectedMin = ArrayImgs.floats(new float[] {
			0, -1, -1, -1,
			0, -1, -1, -1,
			0, -1, -1, -1
		}, 4, 3);
		Img<FloatType> expectedMax = ArrayImgs.floats(new float[] {
			1, 1, 1, 0,
			1, 1, 1, 0,
			1, 1, 1, 0
		}, 4, 3);
		Img<FloatType> expectedMean = ArrayImgs.floats(new float[] {
			1 / 9f, 0, 0, -1 / 9f,
			1 / 9f, 0, 0, -1 / 9f,
			1 / 9f, 0, 0, -1 / 9f
		}, 4, 3);
		Img<FloatType> expectedVariance = ArrayImgs.floats(new float[] {
			1 / 9f, 2 / 8f, 2 / 8f, 1 / 9f,
			1 / 9f, 2 / 8f, 2 / 8f, 1 / 9f,
			1 / 9f, 2 / 8f, 2 / 8f, 1 / 9f
		}, 4, 3);
		feature.apply(new FeatureInput(Views.extendBorder(input), input), RevampUtils.slices(output));
		ImgLib2Assert.assertImageEquals(Views.stack(expectedMin, expectedMax, expectedMean,
			expectedVariance),
			output);
	}

	@Test
	public void testRadius0() {
		OpEnvironment ops = new Context().service(OpService.class);
		FeatureSetting featureSetting = SingleFeatures.statistics(0, true, true, true, true);
		FeatureOp feature = featureSetting.newInstance(ops, GlobalSettings.default2d().build());
		Img<FloatType> input = ArrayImgs.floats(new float[] {
			0, 0, 0, 0,
			0, 1, -1, 0,
			0, 0, 0, 0
		}, 4, 3);
		Img<FloatType> output = ArrayImgs.floats(4, 3, 4);
		Img<FloatType> expectedMin = input;
		Img<FloatType> expectedMax = input;
		Img<FloatType> expectedMean = input;
		Img<FloatType> expectedVariance = ArrayImgs.floats(4, 3);
		feature.apply(new FeatureInput(Views.extendBorder(input), input), RevampUtils.slices(output));
		ImgLib2Assert.assertImageEquals(Views.stack(expectedMin, expectedMax, expectedMean,
			expectedVariance),
			output);
	}

	@Test
	public void testAnisotropic() {
		OpEnvironment ops = new Context().service(OpService.class);
		FeatureSetting featureSetting = SingleFeatures.statistics(1, false, true, false, false);
		FeatureOp feature = featureSetting.newInstance(ops, GlobalSettings.default2d().build());
		Img<FloatType> input = ArrayImgs.floats(new float[] {
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0
		}, 5, 5);
		Img<FloatType> output = ArrayImgs.floats(5, 5, 1);
		Img<FloatType> expectedMax = ArrayImgs.floats(new float[] {
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 1, 1, 1, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0
		}, 5, 5);
		FeatureInput featureInput = new FeatureInput(Views.extendBorder(input), input);
		featureInput.setPixelSize(1.0, 2.0);
		feature.apply(featureInput, RevampUtils.slices(output));
		System.out.println(ToString.toString(output));
		ImgLib2Assert.assertImageEquals(Views.stack(expectedMax), output);

	}
}
