import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.EOFException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

public class DocCollectionInputFormat extends FileInputFormat<LongWritable, Text> {

    private long max_doc = -1;

    public class DocRecordReader extends RecordReader<LongWritable, Text> {
        FSDataInputStream input;
        Text value;
        ArrayList<Integer> al = new ArrayList<>();
        byte[] input_arr;
        byte[] result;
        int doc_num;
        long n_files;
        long start_file;


        @Override
        public void initialize(InputSplit split, TaskAttemptContext context) throws IOException {
            context.getConfiguration();
            FileSplit fsplit = (FileSplit) split;
            Path path = fsplit.getPath();

            String index_file = path.getName();
            index_file = index_file + ".idx";

            FileSystem fs = path.getFileSystem(context.getConfiguration());
            FSDataInputStream input_index = fs.open(new Path(index_file));

            try {
                while (true) {
                    al.add(input_index.readInt());
                }
            } catch (EOFException ignored) {
            }

            long start = fsplit.getStart();

            long offset = 0;
            while (doc_num < start) {
                offset += al.get(doc_num);
                doc_num++;
            }
            n_files = fsplit.getStart();
            start_file = fsplit.getStart();

            if (max_doc < 0)
                throw new IOException("max doc error");

            input = fs.open(path);
            input.seek(offset);

            input_arr = new byte[(int) max_doc];
            result = new byte[(int) max_doc * 20];//?
        }

        @Override
        public boolean nextKeyValue() {
            if (doc_num >= n_files)
                return false;
            Inflater decompresser = new Inflater();
            decompresser.setInput(input_arr, 0, al.get(doc_num));
            try {
                decompresser.inflate(result);
            } catch (DataFormatException e) {
                e.printStackTrace();
            }
            decompresser.end();
            value = new Text(result);
            doc_num++;
            return true;
        }

        @Override
        public LongWritable getCurrentKey() {
            return new LongWritable(al.get(doc_num));
         }

        @Override
        public Text getCurrentValue() {
            return value;
        }

        @Override
        public float getProgress() {
            return (float) (doc_num - start_file) / n_files;
        }

        @Override
        public void close() {
            IOUtils.closeStream(input);
        }
    }

    @Override
    public RecordReader<LongWritable, Text> createRecordReader(InputSplit split, TaskAttemptContext context)
            throws IOException {
        DocRecordReader reader = new DocRecordReader();
        reader.initialize(split, context);
        return reader;
    }

    @Override
    public List<InputSplit> getSplits(JobContext context) throws IOException {
        List<InputSplit> splits = new ArrayList<>();

        for (FileStatus status : listStatus(context)) {
            Path path = status.getPath();

            String index_file = path.getName();

            if (index_file.substring(index_file.length() - 4).equals(".idx")) {
                continue;
            } else {
                index_file = index_file + ".idx";
            }

            FileSystem fs = path.getFileSystem(context.getConfiguration());
            FSDataInputStream input_index = fs.open(new Path(index_file));

            ArrayList<Integer> al = new ArrayList<>();
            try {
                while (true) {
                    int val = input_index.readInt();
                    if (val > max_doc)
                        max_doc = val;
                    al.add(val);
                }
            } catch (EOFException ignored) {
            }

            System.out.println(al);

            int cur_split = 0;
            long split_size = 0;
            long offset = 0;
            for (Integer cur : al) {
                split_size += cur;
                cur_split++;
                long bytes_num_for_split = 100000000;
                if (split_size > bytes_num_for_split) {
                    splits.add(new FileSplit(path, offset, cur_split, null));
                    offset += cur_split;
                    cur_split = 0;
                }
            }
        }
        return splits;
    }
}