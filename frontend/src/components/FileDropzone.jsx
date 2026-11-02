import { useRef, useState } from 'react'

export default function FileDropzone({ onFileSelected, disabled, maxSizeMb = 10 }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleFiles = (files) => {
    if (!disabled && files && files.length > 0) onFileSelected(files[0])
  }

  const openPicker = () => {
    if (!disabled) inputRef.current?.click()
  }

  return (
    <div
      className={`dropzone ${dragging ? 'dragging' : ''}`}
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      onClick={openPicker}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && openPicker()}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        handleFiles(e.dataTransfer.files)
      }}
    >
      <div className="primary-text">Drag and drop a CSV file here</div>
      <div className="secondary-text">or click to browse · CSV only · up to {maxSizeMb} MB</div>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        hidden
        onChange={(e) => {
          handleFiles(e.target.files)
          e.target.value = ''
        }}
      />
    </div>
  )
}
