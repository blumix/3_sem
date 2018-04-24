import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.jobcontrol.JobControl;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;


import java.io.*;
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
                map.put(Long.parseLong(split[1]), Double.parseDouble(split[0]));
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
            urls_with_weights = read_lines( context, "hdfs:/user/m.belozerov/hits/a_scores.txt");
//            urls_with_weights.get(1L);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            Long from = Long.valueOf(value.toString().split("\t")[0]);
            Long to = Long.valueOf(value.toString().split("\t")[1]);
            Double result = urls_with_weights.get(to);
            LongWritable wf = new LongWritable(from);
            context.write(wf, new DoubleWritable(result));
        }
    }

    public static class HitsBMapper extends Mapper<LongWritable, Text, LongWritable, DoubleWritable> {
        Map<Long, Double> urls_with_weights;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            urls_with_weights = read_lines( context, "hdfs:/user/m.belozerov/hits/b_scores.txt");
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

//    private Job getJobConf(String input, String output) throws IOException {
//        Job job = Job.getInstance(getConf());
//
//        job.setJarByClass(GraphBuilder.class);
//        job.setJobName(GraphBuilder.class.getCanonicalName());
//
//        job.setInputFormatClass(TextInputFormat.class);
//        job.setMapOutputValueClass(LongWritable.class);
//        job.setMapOutputKeyClass(LongWritable.class);
//        FileInputFormat.addInputPath(job, new Path(input));
//        FileOutputFormat.setOutputPath(job, new Path(output));
//
//        job.setMapperClass(HitsAMapper.class);
//        job.setReducerClass(HitsReducer.class);
//
//        return job;
//    }

//    public int run(String[] args) throws Exception {
//
//        JobControl jobControl = new JobControl("jobChain");
//        Configuration conf1 = getConf();
//
//        Job job1 = Job.getInstance(conf1);
//        job1.setJarByClass(Hits.class);
//        job1.setJobName("A Combined");
//        job1.setInputFormatClass(TextInputFormat.class);
//
//        FileInputFormat.setInputPaths(job1, new Path(args[0]));
//        FileOutputFormat.setOutputPath(job1, new Path("hdfs:/user/m.belozerov/hits_out/a_scores.txt"));
//
//        job1.setMapperClass(HitsAMapper.class);
//        job1.setReducerClass(HitsReducer.class);
//
//        job1.setOutputKeyClass(LongWritable.class);
//        job1.setOutputValueClass(DoubleWritable.class);
//
//        ControlledJob controlledJob1 = new ControlledJob(conf1);
//        controlledJob1.setJob(job1);
//
//        jobControl.addJob(controlledJob1);
//        Configuration conf2 = getConf();
//
//        Job job2 = Job.getInstance(conf2);
//
//        job2.setJarByClass(Hits.class);
//        job2.setJobName("B combined");
//
//        job2.setInputFormatClass(TextInputFormat.class);
//        FileInputFormat.setInputPaths(job2, new Path(args[0]));
//        FileOutputFormat.setOutputPath(job2, new Path("hdfs:/user/m.belozerov/hits_out/b_scores.txt"));
//
//        job2.setMapperClass(HitsBMapper.class);
//        job2.setReducerClass(HitsReducer.class);
//
//        job2.setOutputKeyClass(LongWritable.class);
//        job2.setOutputValueClass(DoubleWritable.class);
//
//        ControlledJob controlledJob2 = new ControlledJob(conf2);
//        controlledJob2.setJob(job2);
//
//        // make job2 dependent on job1
//        controlledJob2.addDependingJob(controlledJob1);
//        // add the job to the job control
//        jobControl.addJob(controlledJob2);
//        Thread jobControlThread = new Thread(jobControl);
//        jobControlThread.start();
//
//        while (!jobControl.allFinished()) {
//            System.out.println("Jobs in waiting state: " + jobControl.getWaitingJobList().size());
//            System.out.println("Jobs in ready state: " + jobControl.getReadyJobsList().size());
//            System.out.println("Jobs in running state: " + jobControl.getRunningJobList().size());
//            System.out.println("Jobs in success state: " + jobControl.getSuccessfulJobList().size());
//            System.out.println("Jobs in failed state: " + jobControl.getFailedJobList().size());
//            try {
//                Thread.sleep(5000);
//            } catch (Exception ignored) {
//
//            }
//
//        }
//        System.exit(0);
//        return (job1.waitForCompletion(true) ? 0 : 1);
//    }

    private Job getJobConf(String input) throws IOException {
        Job job = Job.getInstance(getConf());

        job.setJarByClass(Hits.class);
        job.setInputFormatClass(TextInputFormat.class);

        FileInputFormat.setInputPaths(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path("hdfs:/user/m.belozerov/hits_out/a_scores.txt"));

        job.setInputFormatClass(TextInputFormat.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setMapOutputKeyClass(LongWritable.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path("hdfs:/user/m.belozerov/hits_out/a_scores_1"));

        job.setMapperClass(HitsAMapper.class);
        job.setReducerClass(HitsReducer.class);

//        job.setNumReduceTasks(10);

        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0]);
        return job.waitForCompletion(true) ? 0 : 1;
    }


//    @Override
//    public int run(String[] args) throws Exception {
//        Job job = getJobConf(args[0], args[1]);
//        return job.waitForCompletion(true) ? 0 : 1;
//    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new Hits(), args);
        System.exit(ret);
    }
}
