package clij;

import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij2.CLIJ2;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.converter.RealTypeConverters;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.trainable_segmention.clij_random_forest.CLIJView;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import org.apache.commons.lang3.ArrayUtils;
import preview.net.imglib2.algorithm.convolution.kernel.Kernel1D;

import java.util.Arrays;
import java.util.HashMap;

public class CLIJKernelConvolution implements NeighborhoodOperation {

	private final CLIJ2 clij;

	private final Kernel1D kernel;

	private final int d;

	public CLIJKernelConvolution(CLIJ2 clij, Kernel1D kernel, int d) {
		this.clij = clij;
		this.kernel = kernel;
		this.d = d;
	}

	@Override
	public Interval getRequiredInputInterval(Interval targetInterval) {
		final long[] min = Intervals.minAsLongArray(targetInterval);
		final long[] max = Intervals.maxAsLongArray(targetInterval);
		min[d] += kernel.min();
		max[d] += kernel.max();
		return new FinalInterval(min, max);
	}

	@Override
	public void convolve(CLIJView input, CLIJView output) {
		final Img<DoubleType> kernelImage = ArrayImgs.doubles(kernel.fullKernel(), kernel.size());
		final ClearCLBuffer kernelBuffer = clij.push(RealTypeConverters.convert(kernelImage, new FloatType()));
		convolve(clij, input, kernelBuffer, output, d);
	}

	public static void convolve(CLIJ2 clij, ClearCLBuffer input, ClearCLBuffer kernel, ClearCLBuffer output) {
		convolve(clij, CLIJView.wrap(input), kernel, CLIJView.wrap(output), 0);
	}

	public static void convolve(CLIJ2 clij, CLIJView input, ClearCLBuffer kernel, CLIJView output, int d) {
		HashMap<String, Object> parameters = new HashMap<>();
		parameters.put("input", input.buffer());
		parameters.put("kernelValues", kernel);
		parameters.put("output", output.buffer());
		HashMap<String, Object> defines = new HashMap<>();
		long[] localSizes = new long[3];
		Arrays.fill(localSizes, 1);
		localSizes[0] = output.interval().dimension(d);
		defines.put("KERNEL_LENGTH", kernel.getWidth());
		defines.put("BLOCK_SIZE", localSizes[0]);
		setSkips(defines, "INPUT", input, d);
		setSkips(defines, "OUTPUT", output, d);
		long[] globalSizes = getDimensions(output.interval());
		ArrayUtils.swap(globalSizes, 0, d);
		clij.executeSubsequently(CLIJKernelConvolution.class, "gauss.cl", "convolve1d", globalSizes,
				globalSizes, localSizes, parameters, defines, null).close();
	}

	private static void setSkips(HashMap<String, Object> defines, String prefix, CLIJView view, int d) {
		ClearCLBuffer buffer = view.buffer();
		Interval interval = view.interval();
		long[] skip = {1, buffer.getWidth(), buffer.getWidth() * buffer.getHeight()};
		defines.put(prefix + "_OFFSET", getOffset(interval, skip));
		ArrayUtils.swap(skip, 0, d);
		defines.put(prefix + "_X_SKIP", skip[0]);
		defines.put(prefix + "_Y_SKIP", skip[1]);
		defines.put(prefix + "_Z_SKIP", skip[2]);
	}

	private static long getOffset(Interval interval, long[] skip) {
		long result = 0;
		for (int i = 0; i < interval.numDimensions(); i++) {
			result += skip[i] * interval.min(i);
		}
		return result;
	}

	private static long[] getDimensions(Interval interval) {
		return new long[]{
				getDimension(interval, 0),
				getDimension(interval, 1),
				getDimension(interval, 2),
		};
	}

	private static long getDimension(Interval interval, int d) {
		return d < interval.numDimensions() ? interval.dimension(d) : 1;
	}
}
