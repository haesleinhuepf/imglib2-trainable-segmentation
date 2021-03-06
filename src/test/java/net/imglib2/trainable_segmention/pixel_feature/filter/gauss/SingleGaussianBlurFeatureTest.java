
package net.imglib2.trainable_segmention.pixel_feature.filter.gauss;

import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.test.ImgLib2Assert;
import net.imglib2.trainable_segmention.Utils;
import net.imglib2.trainable_segmention.pixel_feature.calculator.FeatureCalculator;
import net.imglib2.trainable_segmention.pixel_feature.filter.SingleFeatures;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.Test;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class SingleGaussianBlurFeatureTest {

	private final double sigma = 5.0;

	private final FeatureCalculator calculator = FeatureCalculator.default2d()
		.addFeature(SingleFeatures.gauss(sigma))
		.build();

	@Test
	public void test() {
		RandomAccessible<FloatType> input = Utils.dirac2d();
		RandomAccessibleInterval<FloatType> output = ArrayImgs.floats(5, 5);
		RandomAccessibleInterval<FloatType> expected =
			Utils.create2dImage(output, (x, y) -> Utils.gauss(sigma, x, y));
		calculator.apply(input, Collections.singletonList(output));
		ImgLib2Assert.assertImageEqualsRealType(expected, output, 0.001);
	}

	@Test
	public void testAttributeLabels() {
		List<String> attributes = calculator.attributeLabels();
		List<String> expected = Collections.singletonList("gaussian blur sigma=5.0");
		assertEquals(expected, attributes);
	}
}
