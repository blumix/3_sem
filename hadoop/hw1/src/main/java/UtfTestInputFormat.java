import com.google.common.io.LittleEndianDataInputStream;
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

public class UtfTestInputFormat extends FileInputFormat<LongWritable, Text> {

    private long max_doc = 5000000;

    public class DocRecordReader extends RecordReader<LongWritable, Text> {
        FSDataInputStream input_file;
        String lines;
        Text value;
        boolean returned = false;

        @Override
        public void initialize(InputSplit split, TaskAttemptContext context) throws IOException {
            context.getConfiguration();
            FileSplit fsplit = (FileSplit) split;
            Path path = fsplit.getPath();

            FileSystem fs = path.getFileSystem(context.getConfiguration());
            input_file = fs.open(path);
            lines = input_file.readUTF();
        }

        @Override
        public boolean nextKeyValue() {
            if (returned)
                return false;
            value = new Text(lines);
            returned = true;
            return true;
        }

        @Override
        public LongWritable getCurrentKey() {
            return new LongWritable(1);
        }

        @Override
        public Text getCurrentValue() {
            return value;
        }

        @Override
        public float getProgress() {
            if (returned)
                return 1;
            return 0;
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
        return reader;
    }

    @Override
    public List<InputSplit> getSplits(JobContext context) throws IOException {
        List<InputSplit> splits = new ArrayList<>();

        for (FileStatus status : listStatus(context)) {
            Path path = status.getPath();
            splits.add(new FileSplit(path, 0, 10000, null));
        }
        return splits;
    }
}