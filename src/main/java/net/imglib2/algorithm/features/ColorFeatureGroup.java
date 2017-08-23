package net.imglib2.algorithm.features;

import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.features.ops.FeatureOp;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.real.FloatType;

import java.util.Arrays;
import java.util.List;
import java.util.function.IntPredicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Matthias Arzt
 */
public class ColorFeatureGroup implements FeatureGroup<ARGBType> {

	private final FeatureJoiner joiner;

	ColorFeatureGroup(FeatureOp... features) {
		this(Arrays.asList(features));
	}

	ColorFeatureGroup(List<FeatureOp> features) {
		this.joiner = new FeatureJoiner(features);
	}

	@Override
	public List<FeatureOp> features() {
		return joiner.features();
	}

	@Override
	public int count() {
		return joiner.count() * channelCount();
	}

	@Override
	public void apply(RandomAccessible<ARGBType> input, List<RandomAccessibleInterval<FloatType>> output) {
		List<RandomAccessible<FloatType>> inputs = RevampUtils.splitChannels(input);
		List<List<RandomAccessibleInterval<FloatType>>> outputs = split(output, channelCount());
		for (int i = 0; i < channelCount(); i++)
			joiner.apply(inputs.get(i), outputs.get(i));
	}

	@Override
	public List<String> attributeLabels() {
		return prepend(joiner.globalSettings().imageType().channelNames(), joiner.attributeLabels());
	}

	// -- Helper methods --

	private int channelCount() {
		return 3;
	}

	private static List<String> prepend(List<String> prepend, List<String> labels) {
		return labels.stream()
				.flatMap(label -> prepend.stream().map(pre -> pre.isEmpty() ? label : pre + "_" + label))
				.collect(Collectors.toList());
	}

	private static <T> List<List<T>> split(List<T> input, int count) {
		return IntStream.range(0, count).mapToObj(
				i -> filterByIndexPredicate(input, index -> index % count == i)
		).collect(Collectors.toList());
	}

	private static <T> List<T> filterByIndexPredicate(List<T> in, IntPredicate predicate) {
		return IntStream.range(0, in.size()).filter(predicate).mapToObj(in::get).collect(Collectors.toList());
	}
}
