import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Seo extends Configured implements Tool {
    public static class DocQuestPartitioner extends Partitioner<TextTextPair, IntWritable> {
        @Override
        public int getPartition(TextTextPair key, IntWritable val, int numPartitions) {
            return Math.abs(key.getFirst().hashCode()) % numPartitions;
        }
    }

    public static class KeyComparator extends WritableComparator {
        protected KeyComparator() {
            super(TextTextPair.class, true );
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            return ((TextTextPair)a).compareTo((TextTextPair)b);
        }
    }


    public static class Grouper extends WritableComparator {
        protected Grouper() {
            super(TextTextPair.class, true);
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            Text a_first = ((TextTextPair)a).getFirst();
            Text b_first = ((TextTextPair)b).getFirst();
            return a_first.compareTo(b_first);
        }
    }


    public static class WordCountMapper extends Mapper<LongWritable, Text, TextTextPair, IntWritable> {
        private final static Logger LOG = LoggerFactory.getLogger(WordCountMapper.class);

        static final IntWritable one = new IntWritable(1);
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            String[] input = value.toString().split ("\t");
            String host;
//            LOG.debug(input[1]);
            try {
                host = getDomainName (input[1]);
            } catch (URISyntaxException | NullPointerException e) {
                context.getCounter("COMMON_COUNTERS", "BadURLS").increment(1);
                return;
            }

            TextTextPair composite = new TextTextPair(host, input[0]);
            context.write(composite, one);
        }

        static String getDomainName(String url) throws URISyntaxException , NullPointerException{
            URI uri = new URI(url);
            String domain = uri.getHost();
            return domain.startsWith("www.") ? domain.substring(4) : domain;
        }
    }


    public static final String MIN_CLICKS = "mapreduce.reduce.seo.minclicks";

    public static class WordCountReducer extends Reducer<TextTextPair, IntWritable, Text, IntWritable> {
        @Override
        protected void reduce(TextTextPair key, Iterable<IntWritable> nums, Context context) throws IOException, InterruptedException {

            long min_clicks = context.getConfiguration().getLong(MIN_CLICKS, 1);

            nums.iterator().next().get();
            int most_common_num = 1;
            int current = 1;
            String most_common_quest = key.getSecond ().toString();
            String cur_string = key.getSecond ().toString();
            for(IntWritable ignored : nums) {
                if (!cur_string.equals(key.getSecond ().toString())){
                    if (current > most_common_num){
                        most_common_num = current;
                        most_common_quest = key.getSecond().toString();
                    }
                    cur_string = key.getSecond ().toString();
                    current = 0;
                }
                current++;
            }
            if (most_common_num >= min_clicks)
                context.write(new Text(key.getFirst().toString() +"\t"+ most_common_quest), new IntWritable(most_common_num));
        }
    }

    private Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(getConf());

        job.setJarByClass(Seo.class);
        job.setJobName(Seo.class.getCanonicalName());

        job.setInputFormatClass(TextInputFormat.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        job.setMapperClass(WordCountMapper.class);
        job.setReducerClass(WordCountReducer.class);


        job.setPartitionerClass(DocQuestPartitioner.class);
        job.setSortComparatorClass(KeyComparator.class);
        job.setGroupingComparatorClass(Grouper.class);

        job.setMapOutputKeyClass(TextTextPair.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setNumReduceTasks(4);

        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new Seo(), args);
        System.exit(ret);
    }
}
