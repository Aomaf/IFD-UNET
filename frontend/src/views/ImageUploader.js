import { useEffect, useState } from "react";
import axios from "../axios";
import Dropzone from "react-dropzone";

function ImageUploader() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState();

  return (
    <Dropzone onDrop={(files) => console.log(files)}>
      {({ getRootProps, getInputProps }) => (
        <div className="container">
          <div
            {...getRootProps({
              className: "dropzone",
              onDrop: (event) => event.stopPropagation(),
            })}
          >
            <input {...getInputProps()} />
            <p>Drag 'n' drop some files here, or click to select files</p>
          </div>
        </div>
      )}
    </Dropzone>
  );
}
