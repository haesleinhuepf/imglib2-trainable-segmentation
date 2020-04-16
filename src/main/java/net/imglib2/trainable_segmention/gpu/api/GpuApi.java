package net.imglib2.trainable_segmention.gpu.api;

import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij2.CLIJ2;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.trainable_segmention.RevampUtils;
import net.imglib2.trainable_segmention.utils.Scope;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.util.Intervals;

import java.util.Arrays;
import java.util.HashMap;
import java.util.function.Supplier;

public interface GpuApi extends AutoCloseable {

	static GpuApi getInstance() {
		return Private.getInstance();
	}

	static GpuApi newInstance(String deviceName) {
		return Private.newInstance(deviceName);
	}

	GpuImage create(long[] dimensions, long numberOfChannels, NativeTypeEnum type);

	GpuApi subScope();

	Object exclusive();

	@Override
	void close();

	default GpuImage create(long[] dimensions, NativeTypeEnum type) {
		return handleOutOfMemoryException(() -> {
			return create(dimensions, 1, type);
		});
	}

	default GpuImage push(RandomAccessibleInterval<? extends RealType<?>> source) {
		return handleOutOfMemoryException(() -> {
			GpuImage target = create(Intervals.dimensionsAsLongArray(source), GpuCopy.getNativeTypeEnum(source));
			GpuCopy.copyFromTo(source, target);
			return target;
		});
	}

	default GpuImage pushMultiChannel(RandomAccessibleInterval<? extends RealType<?>> input) {
		return handleOutOfMemoryException(() -> {
			long[] dimensions = Intervals.dimensionsAsLongArray(input);
			int n = dimensions.length - 1;
			GpuImage buffer = create(Arrays.copyOf(dimensions, n), dimensions[n], NativeTypeEnum.Float);
			GpuCopy.copyFromTo(input, buffer);
			return buffer;
		});
	}

	default <T extends RealType<?>> RandomAccessibleInterval<T> pullRAI(GpuImage image) {
		return handleOutOfMemoryException(() -> {
			if(image.getNumberOfChannels() > 1)
				return pullRAIMultiChannel(image);
			return Private.internalPullRai(image, image.getDimensions());
		});
	}

	default <T extends RealType<?>> RandomAccessibleInterval<T> pullRAIMultiChannel(GpuImage image) {
		return handleOutOfMemoryException(() -> {
			return Private.internalPullRai(image, RevampUtils.extend(image.getDimensions(), image.getNumberOfChannels()));
		});
	}

	void execute(Class<?> anchorClass, String kernelFile, String kernelName, long[] globalSizes, long[] localSizes, HashMap<String, Object> parameters, HashMap<String, Object> defines);

	<T> T handleOutOfMemoryException(Supplier<T> action);

	class Private {

		private Private() {
			// prevent from instantiation
		}

		private static <T extends RealType<?>> RandomAccessibleInterval<T> internalPullRai(GpuImage source, long[] dimensions) {
			RealType<?> type = GpuCopy.getImgLib2Type(source.getNativeType());
			Img<T> target = Cast.unchecked(new ArrayImgFactory<>(Cast.unchecked(type)).create(dimensions));
			GpuCopy.copyFromTo(source, target);
			return target;
		}

		private static synchronized GpuApi getInstance() {
			CLIJ2 clij = CLIJ2.getInstance();
			clij.setKeepReferences(false);
			DefaultGpuApi parent = new DefaultGpuApi(clij);
			return new GpuScope(parent, parent);
		}

		public static synchronized GpuApi newInstance(String deviceName) {
			CLIJ2 clij2 = new CLIJ2(new CLIJ(deviceName));
			clij2.setKeepReferences(false);
			DefaultGpuApi parent = new DefaultGpuApi(clij2);
			return new GpuScope(parent, Scope.create(clij2::close, parent));
		}
	}
}
