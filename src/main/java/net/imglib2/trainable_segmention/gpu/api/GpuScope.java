package net.imglib2.trainable_segmention.gpu.api;

import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.imglib2.trainable_segmention.utils.Scope;

import java.util.HashMap;
import java.util.function.Supplier;

public class GpuScope implements GpuApi {

	private final Scope closeQueue = new Scope();

	private final GpuApi parent;

	public GpuScope(GpuApi parent, boolean closeParent) {
		this(parent, closeParent ? parent : null);
	}

	public GpuScope(GpuApi parent, AutoCloseable onClose) {
		this.parent = parent;
		if(onClose != null)
			closeQueue.register(onClose);
	}

	@Override
	public GpuImage create(long[] dimensions, long numberOfChannels, NativeTypeEnum type) {
		return closeQueue.register(parent.create(dimensions, numberOfChannels, type));
	}

	@Override
	public GpuApi subScope() {
		return closeQueue.register(new GpuScope(parent, false));
	}

	@Override
	public Object exclusive() {
		return parent.exclusive();
	}

	@Override
	public void execute(Class<?> anchorClass, String kernelFile, String kernelName, long[] globalSizes, long[] localSizes, HashMap<String, Object> parameters, HashMap<String, Object> defines) {
		parent.execute(anchorClass, kernelFile, kernelName, globalSizes, localSizes, parameters, defines);
	}

	@Override
	public <T> T handleOutOfMemoryException(Supplier<T> action) {
		return parent.handleOutOfMemoryException(action);
	}

	@Override
	public void close() {
		closeQueue.close();
	}
}
