package performance;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.img.CellLoader;
import net.imglib2.cache.img.ReadOnlyCachedCellImgFactory;
import net.imglib2.cache.img.ReadOnlyCachedCellImgOptions;
import net.imglib2.img.Img;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.trainable_segmention.Utils;
import net.imglib2.trainable_segmention.classification.Segmenter;
import net.imglib2.trainable_segmention.gpu.api.GpuApi;
import net.imglib2.trainable_segmention.gson.GsonUtils;
import net.imglib2.trainable_segmention.utils.Scope;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.StopWatch;
import net.imglib2.view.Views;
import org.scijava.Context;
import preview.net.imglib2.parallel.TaskExecutor;
import preview.net.imglib2.parallel.TaskExecutors;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class ParallelSegmentationBenchmark implements Runnable {

	private static final Context context = new Context();

	private static final RandomAccessibleInterval<FloatType> image =
			Utils.loadImageFloatType("/home/arzt/Documents/Datasets/img_TL199_Chgreen.tif");

	private final int index;

	public ParallelSegmentationBenchmark(int index) {
		this.index = index;
	}

	@Override
	public void run() {
		try(Scope scope = new Scope()) {
			GpuApi gpu = scope.register(GpuApi.newInstance(null));

			println("Hello World");
			Segmenter segmenter = Segmenter.fromJson(context, GsonUtils.read("/home/arzt/Documents/Datasets/img_TL199_Chgreen.classifier"));
			println("set use gpu");
			segmenter.setGpu(gpu);
			println("done");
			CellLoader<UnsignedShortType> loader = cell -> segmenter.segment(cell, Views.extendBorder(image));
			int cellSize = 64;
			long imageSize = cellSize * 4;
			int[] cellDims = {cellSize, cellSize, cellSize};
			long[] imageDims = {imageSize, imageSize, imageSize};
			Img<UnsignedShortType> segmentation = createCellImage(loader, imageDims, cellDims);
			List<Interval> cells = getCells(new CellGrid(imageDims, cellDims));
			TaskExecutor executor = TaskExecutors.fixedThreadPool(1);
			StopWatch totalTime = StopWatch.createAndStart();
			executor.forEach(cells, cell -> {
				StopWatch watch = StopWatch.createAndStart();
				RandomAccess<UnsignedShortType> ra = segmentation.randomAccess();
				ra.setPosition(Intervals.minAsLongArray(cell));
				ra.get();
				println(watch);
			});
			totalTime.stop();
			println("Total time: " + totalTime);
			long timePerVoxel = totalTime.nanoTime() / Intervals.numElements(imageDims);
			println("Total time per voxel " + timePerVoxel + " ns");
			println("Total time per pixel " + timePerVoxel * cellSize / 1000 + " us");
		}
	}

	private void println(Object object) {
		System.out.println("[" + index + "] " + object);
	}

	public static void main(String... args) {
		for (int i = 1; i <= 4; i++) {
			startThread(i);
		}
	}

	private static void startThread(int i) {
		new Thread(new ParallelSegmentationBenchmark(i), "my-thread-" + i).start();
	}

	private static List<Interval> getCells(CellGrid grid) {
		List<Interval> cells = new ArrayList<>();
		long numCells = Intervals.numElements(grid.getGridDimensions());
		for (int i = 0; i < numCells; i++) {
			long[] cellMin = new long[3];
			int[] cellDims = new int[3];
			grid.getCellDimensions(i, cellMin, cellDims);
			cells.add( FinalInterval.createMinSize(cellMin, IntStream.of(cellDims).mapToLong(x -> x).toArray()));
		}
		return cells;
	}

	private static Img<UnsignedShortType> createCellImage(CellLoader<UnsignedShortType> loader, long[] imageSize, int[] cellSize) {
		ReadOnlyCachedCellImgFactory factory = new ReadOnlyCachedCellImgFactory( new ReadOnlyCachedCellImgOptions().cellDimensions(cellSize));
		return factory.create(imageSize, new UnsignedShortType(), loader);
	}
}
