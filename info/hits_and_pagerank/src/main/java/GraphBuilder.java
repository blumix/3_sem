import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.*;
import java.util.*;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;


class LinksExtractor {
    private Map<String, Integer> ids;
    Integer current_id = -1;
    private ArrayList<String> cur_links;

    LinksExtractor(JobContext context) {
        ids = get_ids(context);
    }


    private Map<String, Integer> get_ids(JobContext context) {
        Map<String, Integer> map = new HashMap<>();

        try {
            Path pt=new Path("hdfs:/data/infopoisk/hits_pagerank/urls.txt");//Location of file in HDFS
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader bufferedReader =new BufferedReader(new InputStreamReader(fs.open(pt)));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                String[] split = line.split("\t");
                map.put(split[1], Integer.parseInt(split[0]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return map;
    }

    private void getLinks(String html) {
        Document doc = Jsoup.parse(html);
        Elements links = doc.select("a[href]");

        cur_links = new ArrayList<>();
        for (Element link : links) {
            String ext_link = link.attr("href");
            if (ext_link.charAt(0) == '/') {
                ext_link = "http://lenta.ru" + ext_link;
            }
            cur_links.add(ext_link);
        }
    }

    private static String unzip(String encoded) {
        byte[] compressed;
        try {
            compressed = Base64.getMimeDecoder().decode(encoded);
        } catch (Exception e) {
            throw new RuntimeException("Failed to unzip content", e);
        }

        if ((compressed == null) || (compressed.length == 0)) {
            throw new IllegalArgumentException("Cannot unzip null or empty bytes");
        }
        try {
            Inflater inflater = new Inflater();
            inflater.setInput(compressed, 0, compressed.length);
            byte[] result = new byte[1000000];
            int resultLength = inflater.inflate(result);
            inflater.end();
            return new String(result, 0, resultLength, "UTF-8");
        } catch (DataFormatException | UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        return "";
    }


    private ArrayList<Integer> get_connections() {
        ArrayList<Integer> links = new ArrayList<>();
        for (String link : cur_links) {
            if (ids.containsKey(link)) {
                links.add(ids.get(link));
            }
        }
        return links;

    }

    ArrayList<Integer> go_parse(String compr_html) {
        compr_html = compr_html.trim();
        String[] splited = compr_html.split("\t");
        current_id = Integer.parseInt(splited[0]);
        String html = unzip(splited[1]);
        getLinks(html);
        return get_connections();
    }
}

public class GraphBuilder extends Configured implements Tool {
    public class GraphBuilderMapper extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
        LinksExtractor linksExtractor;
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            linksExtractor = new LinksExtractor(context);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            ArrayList<Integer> links = linksExtractor.go_parse(value.toString());

            for (int link : links) {
                context.write(new IntWritable(linksExtractor.current_id), new IntWritable(link));
            }
        }

    }

    public static class GraphBuilderReducer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
        @Override
        protected void reduce(IntWritable key, Iterable<IntWritable> nums, Context context) throws IOException, InterruptedException {

            for (IntWritable val : nums) {
                context.write(key, val);
            }
        }
    }

    private Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(getConf());

        job.setJarByClass(GraphBuilder.class);
        job.setJobName(GraphBuilder.class.getCanonicalName());

        job.setInputFormatClass(TextInputFormat.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        job.setMapperClass(GraphBuilderMapper.class);
        job.setReducerClass(GraphBuilderReducer.class);

        job.setNumReduceTasks(10);

        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new GraphBuilder(), args);
        System.exit(ret);
    }
}
