package net.imglib2.trainable_segmention.gpu.algorithms;

import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.Duplicator;
import ij.plugin.GaussianBlur3D;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij2.CLIJ2;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.test.ImgLib2Assert;
import net.imglib2.trainable_segmention.RevampUtils;
import net.imglib2.trainable_segmention.Utils;
import net.imglib2.trainable_segmention.gpu.api.GpuApi;
import net.imglib2.trainable_segmention.gpu.api.GpuImage;
import net.imglib2.trainable_segmention.gpu.api.GpuViews;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.jruby.RubyProcess;
import preview.net.imglib2.algorithm.gauss3.Gauss3;

import java.util.Arrays;

public class HaasePlayground {
    public static void main(String[] args) {

        double sigma = 20;
        int iterations = 10;

        GpuApi gpu = GpuApi.getInstance();
        RandomAccessible<FloatType> dirac = Utils.dirac(3);
        RandomAccessibleInterval<FloatType> cpuResult = RevampUtils.createImage(Intervals.createMinMax(-2, -2, -2, 2, 2, 2), new FloatType());
        for (int i = 0; i < iterations; i++) {
            long time = System.nanoTime();
            Gauss3.gauss(sigma, dirac, cpuResult);
            System.out.println("Gauss3 took " + ((System.nanoTime() - time)/1000000));
        }
        Interval targetInterval = new FinalInterval(cpuResult);
        GpuNeighborhoodOperation operation = GpuGauss.gauss(gpu, sigma, sigma, sigma);
        Interval inputInterval = operation.getRequiredInputInterval(targetInterval);

        RandomAccessibleInterval<FloatType> gpuResult = null;
        for (int i = 0; i < iterations; i++) {
            long time = System.nanoTime();
            try (
                    GpuImage input = gpu.push(Views.interval(dirac, inputInterval));
                    GpuImage output = gpu.create(Intervals.dimensionsAsLongArray(targetInterval), NativeTypeEnum.Float);
            ) {
                operation.apply(GpuViews.wrap(input), GpuViews.wrap(output));
                gpuResult = gpu.pullRAI(output);
            }
            System.out.println("GpuApi took " + ((System.nanoTime() - time)/1000000));
        }
        ImgLib2Assert.assertImageEqualsRealType(Views.zeroMin(cpuResult), gpuResult, 1.e-7);

        System.out.println(inputInterval);

        RandomAccessibleInterval clijResult = null;
        ImagePlus ijResult = null;
        {
            CLIJ2 clij2 = CLIJ2.getInstance();
            RandomAccessibleInterval input = Views.interval(dirac, inputInterval);

            for (int i = 0; i < iterations; i++) {
                long time = System.nanoTime();
                ClearCLBuffer clijInput = clij2.push(input);
                ClearCLBuffer clijOutput = clij2.create(clijInput);

                clij2.gaussianBlur3D(clijInput, clijOutput, sigma, sigma, sigma);
                clijResult = clij2.getCLIJ().pullRAI(clijOutput);
                System.out.println("CLIJ2 RAI took " + ((System.nanoTime() - time)/1000000));
            }

            ImagePlus imp = clij2.pull(clij2.push(input));
            System.out.println(imp.getNSlices());
            for (int i = 0; i < iterations; i++) {
                long time = System.nanoTime();
                ClearCLBuffer clijInput = clij2.push(imp);
                ClearCLBuffer clijOutput = clij2.create(clijInput);

                clij2.gaussianBlur3D(clijInput, clijOutput, sigma, sigma, sigma);
                clij2.getCLIJ().pull(clijOutput);
                System.out.println("CLIJ2 IMP took " + ((System.nanoTime() - time)/1000000));
            }

            for (int i = 0; i < iterations; i++) {
                long time = System.nanoTime();
                ijResult = new Duplicator().run(imp, 1, imp.getNSlices());
                GaussianBlur3D.blur(ijResult, sigma, sigma, sigma);
                System.out.println("IJ IMP took " + ((System.nanoTime() - time)/1000000));
            }


        }

        // analyse / show results
        {

            System.out.println("cpuResult: " + cpuResult);
            System.out.println("gpuResult: " + gpuResult);
            System.out.println("clijResult: " + clijResult);
            System.out.println("ijResult: " + ijResult);

            CLIJ2 clij2 = CLIJ2.getInstance();
            ClearCLBuffer res1 = clij2.push(cpuResult);
            ClearCLBuffer res2 = clij2.push(gpuResult);
            ClearCLBuffer res3 = clij2.push(clijResult);
            ClearCLBuffer res4 = clij2.push(ijResult);

            System.out.println("Size1: " + Arrays.toString(res1.getDimensions()));
            System.out.println("Size2: " + Arrays.toString(res2.getDimensions()));
            System.out.println("Size3: " + Arrays.toString(res3.getDimensions()));
            System.out.println("Size3: " + Arrays.toString(res4.getDimensions()));

            System.out.println("MSE12: " + clij2.meanSquaredError(res1, res2));
            System.out.println("MSE23: " + clij2.meanSquaredError(res2, res3));
            System.out.println("MSE13: " + clij2.meanSquaredError(res1, res3));
            System.out.println("MSE14: " + clij2.meanSquaredError(res1, res4));
            System.out.println("MSE24: " + clij2.meanSquaredError(res2, res4));
            System.out.println("MSE34: " + clij2.meanSquaredError(res3, res4));

            new ImageJ();
            clij2.show(res1, "res1");
            clij2.show(res2, "res2");
            clij2.show(res3, "res3");
        }
    }
}
