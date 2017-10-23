package net.imglib2.trainable_segmention.ops;

import net.imagej.ops.Op;
import net.imagej.ops.special.function.UnaryFunctionOp;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmention.GlobalSettings;
import net.imglib2.type.numeric.real.FloatType;
import org.scijava.plugin.SciJavaPlugin;

import java.util.List;

/**
 * @author Matthias Arzt
 */
public interface FeatureOp extends SciJavaPlugin, Op, UnaryFunctionOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> {

	int count();

	List<String> attributeLabels();

	void apply(RandomAccessible<FloatType> input, List<RandomAccessibleInterval<FloatType>> output);

	GlobalSettings globalSettings();
}
