import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;

public class GetResultPaRa extends Configured implements Tool {
    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new GetResultPaRa(), args);
        System.exit(ret);
    }

    private Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(getConf());

        job.setJarByClass(GetResultPaRa.class);
        job.setJobName(GetResultPaRa.class.getCanonicalName());

        job.setInputFormatClass(TextInputFormat.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setMapOutputKeyClass(LongWritable.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        job.setMapperClass(ResMapper.class);
        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static class ResMapper extends Mapper<LongWritable, Text, DoubleWritable, Text> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            int tIdx1 = value.find("\t");
            int tIdx2 = value.find("\t", tIdx1 + 1);

            String page = Text.decode(value.getBytes(), 0, tIdx1);
            float pageRank = Float.parseFloat(Text.decode(value.getBytes(), tIdx1 + 1, tIdx2 - (tIdx1 + 1)));

            context.write(new DoubleWritable(pageRank), new Text(page));
        }
    }
}
