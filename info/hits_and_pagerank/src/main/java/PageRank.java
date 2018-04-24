import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;


public class PageRank {
    // utility attributes
    private static final String LINKS_SEPARATOR = "|";
    private static final Double DAMPING = 0.85;
    private static NumberFormat NF = new DecimalFormat("00");
    private static int ITERATIONS = 2;


    public static void main(String[] args) throws Exception {

        PageRank pagerank = new PageRank();

        String OUT_PATH = "OutPaRa";
        
        for (int runs = 0; runs < ITERATIONS; runs++) {
            String inPath = OUT_PATH + "/iter" + NF.format(runs);
            String lastOutPath = OUT_PATH + "/iter" + NF.format(runs + 1);
            System.out.println("Running Job#2 [" + (runs + 1) + "/" + PageRank.ITERATIONS + "] (PageRank calculation) ...");
            boolean isCompleted = pagerank.job(inPath, lastOutPath);
            if (!isCompleted) {
                System.exit(1);
            }
        }
        System.out.println("DONE!");
        System.exit(0);
    }

    private boolean job(String in, String out) throws IOException,
            ClassNotFoundException,
            InterruptedException {

        Job job = Job.getInstance(new Configuration());
        job.setJarByClass(PageRank.class);

        FileInputFormat.setInputPaths(job, new Path(in));
        job.setInputFormatClass(TextInputFormat.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setMapperClass(PageRankMapper.class);

        FileOutputFormat.setOutputPath(job, new Path(out));
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setReducerClass(PageRankJob2Reducer.class);

        return job.waitForCompletion(true);

    }


    public static class PageRankMapper extends Mapper<LongWritable, Text, Text, Text> {

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            /* PageRank calculation algorithm (mapper)
             * Input file format (separator is TAB):
             *
             *     <title>    <page-rank>    <link1>,<link2>,<link3>,<link4>,... ,<linkN>
             *
             * Output has 2 kind of records:
             * One record composed by the collection of links of each page:
             *
             *     <title>   |<link1>,<link2>,<link3>,<link4>, ... , <linkN>
             *
             * Another record composed by the linked page, the page rank of the source page
             * and the total amount of out links of the source page:
             *
             *     <link>    <page-rank>    <total-links>
             */

            int tIdx1 = value.find("\t");
            int tIdx2 = value.find("\t", tIdx1 + 1);

            // extract tokens from the current line
            String page = Text.decode(value.getBytes(), 0, tIdx1);
            String pageRank = Text.decode(value.getBytes(), tIdx1 + 1, tIdx2 - (tIdx1 + 1));
            String links = Text.decode(value.getBytes(), tIdx2 + 1, value.getLength() - (tIdx2 + 1));

            String[] allOtherPages = links.split(",");
            for (String otherPage : allOtherPages) {
                Text pageRankWithTotalLinks = new Text(pageRank + "\t" + allOtherPages.length);
                context.write(new Text(otherPage), pageRankWithTotalLinks);
            }

            // put the original links so the reducer is able to produce the correct output
            context.write(new Text(page), new Text(PageRank.LINKS_SEPARATOR + links));

        }

    }


    public static class PageRankJob2Reducer extends Reducer<Text, Text, Text, Text> {

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
                InterruptedException {

            /* PageRank calculation algorithm (reducer)
             * Input file format has 2 kind of records (separator is TAB):
             *
             * One record composed by the collection of links of each page:
             *
             *     <title>   |<link1>,<link2>,<link3>,<link4>, ... , <linkN>
             *
             * Another record composed by the linked page, the page rank of the source page
             * and the total amount of out links of the source page:
             *
             *     <link>    <page-rank>    <total-links>
             */

            StringBuilder links = new StringBuilder();
            double sumShareOtherPageRanks = 0.0;

            for (Text value : values) {

                String content = value.toString();

                if (content.startsWith(PageRank.LINKS_SEPARATOR)) {
                    // if this value contains node links append them to the 'links' string
                    // for future use: this is needed to reconstruct the input for Job#2 mapper
                    // in case of multiple iterations of it.
                    links.append(content.substring(PageRank.LINKS_SEPARATOR.length()));
                } else {

                    String[] split = content.split("\\t");

                    // extract tokens
                    double pageRank = Double.parseDouble(split[0]);
                    int totalLinks = Integer.parseInt(split[1]);

                    // add the contribution of all the pages having an outlink pointing 
                    // to the current node: we will add the DAMPING factor later when recomputing
                    // the final pagerank value before submitting the result to the next job.
                    sumShareOtherPageRanks += (pageRank / totalLinks);
                }

            }

            double newRank = PageRank.DAMPING * sumShareOtherPageRanks + (1 - PageRank.DAMPING);
            context.write(key, new Text(newRank + "\t" + links));

        }

    }
}