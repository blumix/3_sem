import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;


public class Hits extends Configured implements Tool {

    private static Map<Long, Double> read_lines(Mapper.Context context, String filename) throws IOException {
        Map<Long, Double> map = new HashMap<>();

//        try {
            Path pt=new Path(filename);//Location of file in HDFS
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fs.open(pt)));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                String[] split = line.split("\t");
                map.put(Long.parseLong(split[0]), Double.parseDouble(split[1]));
            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        return map;
    }


    public static class HitsAMapper extends Mapper<LongWritable, Text, LongWritable, DoubleWritable> {
        Map<Long, Double> urls_with_weights;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            Configuration conf = context.getConfiguration();
            String param = conf.get("epoch");
            String fname = "hdfs:/user/m.belozerov/hits_out_" + param + "/a_scores.txt";
            urls_with_weights = read_lines( context, fname);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            Long from = Long.valueOf(value.toString().split("\t")[0]);
            Long to = Long.valueOf(value.toString().split("\t")[1]);

            context.write(new LongWritable(from), new DoubleWritable(urls_with_weights.get(to)));
        }
    }

    public static class HitsBMapper extends Mapper<LongWritable, Text, LongWritable, DoubleWritable> {
        Map<Long, Double> urls_with_weights;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            Configuration conf = context.getConfiguration();
            String param = conf.get("epoch");
            String fname = "hdfs:/user/m.belozerov/hits_out_" + param + "/b_scores.txt";
            urls_with_weights = read_lines( context, fname);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            Long from = key.get();
            Long to = Long.valueOf(value.toString());

            context.write(new LongWritable(to), new DoubleWritable(urls_with_weights.get(from)));
        }
    }


    public static class HitsReducer extends Reducer<LongWritable, DoubleWritable, LongWritable, DoubleWritable> {
        @Override
        protected void reduce(LongWritable key, Iterable<DoubleWritable> nums, Context context) throws IOException, InterruptedException {

            Double sum = 0.;
            for (DoubleWritable val : nums) {
                sum += val.get();
            }

            context.write(key, new DoubleWritable(sum));
        }
    }


    private Job getJobConf_A(String input, Integer epoch) throws IOException {

        getConf().set("epoch", String.valueOf(epoch));
        Job job = Job.getInstance(getConf());

        job.setJarByClass(Hits.class);
        job.setInputFormatClass(TextInputFormat.class);

        FileInputFormat.setInputPaths(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path("hdfs:/user/m.belozerov/hits_out_" + String.valueOf(epoch) + "/a_scores.txt"));

        job.setInputFormatClass(TextInputFormat.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setMapOutputKeyClass(LongWritable.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path("hdfs:/user/m.belozerov/hits_out_"+ String.valueOf(epoch + 1)+ "/b_scores"));

        job.setMapperClass(HitsAMapper.class);
        job.setReducerClass(HitsReducer.class);

        return job;
    }

    private Job getJobConf_B(String input, Integer epoch) throws IOException {

        getConf().set("epoch", String.valueOf(epoch));
        Job job = Job.getInstance(getConf());

        job.setJarByClass(Hits.class);
        job.setInputFormatClass(TextInputFormat.class);

        FileInputFormat.setInputPaths(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path("hdfs:/user/m.belozerov/hits_out_" + String.valueOf(epoch) + "/b_scores.txt"));

        job.setInputFormatClass(TextInputFormat.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setMapOutputKeyClass(LongWritable.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path("hdfs:/user/m.belozerov/hits_out_"+ String.valueOf(epoch + 1)+ "/a_scores"));

        job.setMapperClass(HitsBMapper.class);
        job.setReducerClass(HitsReducer.class);

        return job;
    }

    @Override
    public int run(String[] args) throws Exception {

        int res = 0;
        for (int epoch = 1; epoch < 5; epoch++) {
            {
                Job job = getJobConf_A(args[0], epoch);
                res += job.waitForCompletion(true) ? 0 : 1;
            }
            {
                Job job = getJobConf_B(args[0], epoch);
                res += job.waitForCompletion(true) ? 0 : 1;
            }
        }
        return res;
    }


    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new Hits(), args);
        System.exit(ret);
    }
}
