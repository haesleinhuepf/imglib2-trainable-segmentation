package net.imglib2.trainable_segmention.gpu.api;

import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.PooledObjectFactory;
import org.apache.commons.pool2.impl.DefaultPooledObject;
import org.apache.commons.pool2.impl.GenericObjectPool;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;
import org.junit.Test;

public class PoolTest {

	@Test
	public void test() throws Exception {
		GenericObjectPoolConfig<MyObject> config = new GenericObjectPoolConfig<>();
		config.setMaxTotal(2);
		config.setMinIdle(0);
		config.setMinEvictableIdleTimeMillis(2000);
		config.setTimeBetweenEvictionRunsMillis(500);
		GenericObjectPool<MyObject> pool = new GenericObjectPool<>(new MyObjectFactory(), config);
		MyObject a = pool.borrowObject();
		pool.returnObject(a);
		MyObject b = pool.borrowObject();
		pool.returnObject(b);
		Thread.sleep(10000);
		MyObject c = pool.borrowObject();
		pool.returnObject(c);
		pool.close();

	}

	private static class MyObjectFactory implements PooledObjectFactory<MyObject> {

		@Override
		public PooledObject<MyObject> makeObject() throws Exception {
			return new DefaultPooledObject<>(new MyObject());
		}

		@Override
		public void destroyObject(PooledObject<MyObject> pooledObject) {
			pooledObject.getObject().close();
		}

		@Override
		public boolean validateObject(PooledObject<MyObject> pooledObject) {
			return true;
		}

		@Override
		public void activateObject(PooledObject<MyObject> pooledObject) {

		}

		@Override
		public void passivateObject(PooledObject<MyObject> pooledObject) {
			pooledObject.getObject().clear();
		}
	}

	private static class MyObject {

		public MyObject() {
			System.out.println("create");
		}

		public void clear() {
			System.out.println("clear");
		}

		public void close() {
			System.out.println("close");
		}
	}
}
