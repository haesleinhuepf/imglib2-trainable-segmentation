package net.imglib2.trainable_segmention.gpu.api;

import net.haesleinhuepf.clij.clearcl.exceptions.OpenCLException;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij2.CLIJ2;

import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.function.Supplier;

public class DefaultGpuApi implements GpuApi {

	private final CLIJ2 clij;

	private final ClearCLBufferPool pool;

	private final static Set<ClearCLBufferPool> pools = new CopyOnWriteArraySet<>();

	public DefaultGpuApi(CLIJ2 clij) {
		this.clij = clij;
		this.pool = new ClearCLBufferPool(clij.getCLIJ().getClearCLContext());
		pools.add(pool);
	}

	@Override
	public GpuImage create(long[] dimensions, long numberOfChannels, NativeTypeEnum type) {
		return handleOutOfMemoryException( () -> new GpuImage(pool.create(dimensions, numberOfChannels, type), pool::release));
	}

	@Override
	public GpuApi subScope() {
		return new GpuScope(this, false);
	}

	@Override
	public Object exclusive() {
		return clij;
	}

	@Override
	public void close() {
		pools.remove(pool);
		pool.close();
	}

	@Override
	public void execute(Class<?> anchorClass, String kernelFile, String kernelName, long[] globalSizes, long[] localSizes, HashMap<String, Object> parameters, HashMap<String, Object> defines) {
		for(String key : parameters.keySet()) {
			Object value = parameters.get(key);
			if(value instanceof GpuImage)
				parameters.put(key, ((GpuImage) value).clearCLBuffer());
		}
		handleOutOfMemoryException(() -> {
			clij.executeSubsequently(anchorClass, kernelFile, kernelName, null, globalSizes, localSizes, parameters, defines, null).close();
			return null;
		});
	}

	@Override
	public <T> T handleOutOfMemoryException(Supplier<T> action) {
		try {
			return action.get();
		} catch (OpenCLException exception) {
			if(exception.getErrorCode() == -4) {
				System.err.println("*** GPU memory, clear garbage ***");
				pools.forEach(ClearCLBufferPool::clear);
				return action.get();
			}
			else
				throw exception;
		}
	}

}
