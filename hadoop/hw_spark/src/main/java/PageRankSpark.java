import com.google.common.collect.Lists;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;


import java.util.ArrayList;
import java.util.List;

public class PageRankSpark {
    public static void main(String[] args) {

        if (args.length != 3)
            System.out.println("Usage: iterations, alpha, input, output");

        int iterations = Integer.valueOf(args[0]);
        String inputFile = args[2];
        String outputFile = args[3];

        SparkConf conf = new SparkConf().setAppName("PageRank");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> input_f = sc.textFile(inputFile);
        JavaRDD<String> input = input_f.filter(str -> !(str.charAt(0) == '#'));

        JavaPairRDD<Long, Long> pairs = input.mapToPair(v -> {
            String[] pair = v.split("\t");
            return new Tuple2<>(Long.valueOf(pair[0]), Long.valueOf(pair[1]));
        });

        JavaPairRDD<Long, ArrayList<Long>> links = pairs
                .groupByKey()
                .mapToPair(k -> new Tuple2<>(k._1(), Lists.newArrayList(k._2())))
                .cache();

        long count = links.count();
        double singlePR = 1. / count;
        double alpha = Double.valueOf(args[1]);
        long no_out_count = links.filter(K-> K._2().isEmpty()).count();
        JavaPairRDD<Long, Double> pr = links.mapToPair(k -> new Tuple2<>(k._1(), singlePR));

        for (int i = 0; i < iterations; i++) {
            double no_out_sum = pr.join(links).filter(K-> K._2()._2().isEmpty()).map(K-> K._2()._1()).reduce ((K, V) -> K += V);
            double add_pr = no_out_sum / no_out_count;
            pr = pr.join(links)
                    .flatMapToPair(k -> {
                        ArrayList<Tuple2<Long, Double>> arr = new ArrayList<>();
                        int len = k._2()._2().size();
                        for (Long n : k._2()._2()) {
                            arr.add(new Tuple2<>(n, k._2()._1() / len));
                        }
                        return arr.iterator();
                    })
                    .reduceByKey((K, V) -> {
                        if (K == 0) {
                            K += singlePR * alpha + (1-alpha) * add_pr;
                        }
                        return K + V * (1 - alpha);
                    });
        }

        JavaPairRDD<Double, Long> sorted_pr = pr.mapToPair(Tuple2::swap).sortByKey();
        sorted_pr.saveAsTextFile(outputFile);
    }
}