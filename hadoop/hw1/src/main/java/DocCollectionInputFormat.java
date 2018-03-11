import com.google.common.io.LittleEndianDataInputStream;
import org.apache.commons.configuration.Configuration;
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

    private long max_doc = 5000000;

    public class DocRecordReader extends RecordReader<LongWritable, Text> {
        FSDataInputStream input_file;
        Text value;
        List<Integer> index_array;
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
            FSDataInputStream input_index = fs.open(new Path(path.getParent(), index_file));

            prepare_index(fsplit, path, fs, input_index);

        }

        private void prepare_index(FileSplit fsplit, Path path, FileSystem fs, FSDataInputStream input_index) throws IOException {
            index_array = read_index(input_index);
            start_file = fsplit.getStart();

            long offset = 0;
            while (doc_num < start_file) {
                offset += index_array.get(doc_num);
                doc_num++;
            }
            n_files = fsplit.getLength();

            if (max_doc < 0)
                throw new IOException("max doc error");

            input_file = fs.open(path);
            input_file.seek(offset);

            input_arr = new byte[(int) max_doc];
            result = new byte[(int) max_doc * 20];//?
        }

        @Override
        public boolean nextKeyValue() throws IOException {
            if (doc_num >= n_files)
                return false;

//            System.out.println("Doc num:" + doc_num + ", N files:" + n_files);
            try {

                input_file.readFully(input_arr, 0, index_array.get(doc_num));
            } catch (IOException e) {
                e.printStackTrace();
            }
            Inflater decompresser = new Inflater();
            decompresser.setInput(input_arr, 0, index_array.get(doc_num));
            int res_len = 0;
            try {
                if ((res_len = decompresser.inflate(result)) > 150000 * 5)
                    System.out.println("decompress error");
            } catch (DataFormatException e) {
                e.printStackTrace();
            }
            decompresser.end();
            value = new Text (new String(result, 0, res_len, "UTF-8"));
            doc_num++;
            return true;
        }

        @Override
        public LongWritable getCurrentKey() {
            return new LongWritable(index_array.get(doc_num - 1));
         }

        @Override
        public Text getCurrentValue() throws IOException {
//            System.out.println(value);
//            throw new IOException(value.toString());
            return value;
        }

        @Override
        public float getProgress() {
            return (float) (doc_num ) / n_files;
        }

        @Override
        public void close() {
            IOUtils.closeStream(input_file);
        }

    }

    @Override
    public RecordReader<LongWritable, Text> createRecordReader(InputSplit split, TaskAttemptContext context)
            throws IOException {
        DocRecordReader reader = new DocRecordReader();
        reader.initialize(split, context);
//        System.out.println("Here?");
        return reader;
    }

    private static List<Integer> read_index(FSDataInputStream index_file) throws IOException {
        int max_doc = 0;
        LittleEndianDataInputStream in = new LittleEndianDataInputStream(index_file);
        List<Integer> al = new ArrayList<>();
        try {
            while (true){
                int val = in.readInt();
                if (val > max_doc)
                    max_doc = val;
                al.add(val);
            }
        } catch (EOFException ignored) {
        }
        return al;
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
            FSDataInputStream input_index = fs.open(new Path(path.getParent(), index_file));
            List<Integer> al = read_index(input_index);

            int cur_split = 0;
            long split_size = 0;
            long offset = 0;
            for (Integer cur : al) {
                split_size += cur;
                if (cur > max_doc)
                    max_doc = cur;
                cur_split++;
                long bytes_num_for_split = getNumBytesPerSplit (context.getConfiguration());
                if (split_size > bytes_num_for_split) {
                    splits.add(new FileSplit(path, offset, cur_split, null));
                    offset += cur_split;
                    split_size = 0;
                    cur_split = 0;
                }
            }
            splits.add(new FileSplit(path, offset, cur_split, null));
        }
        return splits;
    }
    public static final String BYTES_PER_MAP = "mapreduce.input.bmp.bytes_per_map";

    public static long getNumBytesPerSplit(Configuration conf) {
        return  conf.getLong(BYTES_PER_MAP, 134217728);
    }

}