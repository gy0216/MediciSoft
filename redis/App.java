package jedis.project.jedisTest2;

import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

public class App 
{
    public static void main( String[] args )
    {

    	JedisPool pool = new JedisPool(new JedisPoolConfig(), "localhost");
    	
    	Jedis jedis = null;
    	try {
    	  jedis = pool.getResource();
		  jedis.set("foo", "bar");
    	  String foobar = jedis.get("foo");
    	  System.out.println(foobar);
    	} finally {
//    	  
    	  if (jedis != null) {
    	    jedis.close();
    	  }
    	}
    	/// ... when closing your application:
    	pool.close();
    }
}
