package net.imglib2.trainable_segmention.gpu.api;

import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij2.CLIJ2;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.PooledObjectFactory;
import org.apache.commons.pool2.impl.DefaultPooledObject;
import org.apache.commons.pool2.impl.GenericObjectPool;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;

public class GpuPool {

	private final GenericObjectPool<DefaultGpuApi> pool;

	private static final GpuPool POOL = new GpuPool();

	public static GpuApi borrowGpu() {
		return POOL.gpu();
	}

	private GpuPool() {
		GenericObjectPoolConfig<DefaultGpuApi> config = new GenericObjectPoolConfig<>();
		config.setMaxTotal(4);
		config.setMinIdle(0);
		config.setMinEvictableIdleTimeMillis(2000);
		config.setTimeBetweenEvictionRunsMillis(500);
		this.pool = new GenericObjectPool<>(new MyObjectFactory(), config);
		Runtime.getRuntime().addShutdownHook(new Thread(() -> pool.close()));
	}

	public GpuApi gpu() {
		try {
			DefaultGpuApi gpu = pool.borrowObject();
			return new GpuScope(gpu, () -> {
				// this is executed when the GpuScope is closed
				pool.returnObject(gpu);
			});
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static synchronized CLIJ2 createCLIJ2() {
		return new CLIJ2(new CLIJ(null));
	}

	private static class MyObjectFactory implements PooledObjectFactory<DefaultGpuApi> {

		@Override
		public PooledObject<DefaultGpuApi> makeObject() throws Exception {
			CLIJ2 clij2 = createCLIJ2();
			DefaultGpuApi defaultGpuApi = new DefaultGpuApi(clij2);
			return new DefaultPooledObject<>(defaultGpuApi);
		}

		@Override
		public void destroyObject(PooledObject<DefaultGpuApi> pooledObject) {
			pooledObject.getObject().close();
		}

		@Override
		public boolean validateObject(PooledObject<DefaultGpuApi> pooledObject) {
			return true;
		}

		@Override
		public void activateObject(PooledObject<DefaultGpuApi> pooledObject) {

		}

		@Override
		public void passivateObject(PooledObject<DefaultGpuApi> pooledObject) {

		}
	}
}
